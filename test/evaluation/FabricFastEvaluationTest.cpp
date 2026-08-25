#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Evaluation/ModelParameter.h"
#include "Evaluation/ModelParameterBundle.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/Models/CalibratedFpa.h"
#include "Evaluation/Models/CanonicalDataflowFabricAnalytic.h"
#include "Evaluation/Models/FabricLowConfidence.h"
#include "Evaluation/Models/FpaParameterContract.h"
#include "Evaluation/ProductionRegistry.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using namespace loom;
using namespace loom::evaluation;

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "fabric fast evaluation test failed: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-fabric-fast-evaluation", path_))
      fail(error.message());
  }
  ~TemporaryDirectory() { llvm::sys::fs::remove_directories(path_); }
  llvm::StringRef path() const { return path_; }

private:
  llvm::SmallString<128> path_;
};

mlir::MLIRContext &context() {
  static mlir::MLIRContext *value = [] {
    mlir::DialectRegistry registry;
    registry.insert<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                    mlir::arith::ArithDialect, mlir::DLTIDialect,
                    mlir::func::FuncDialect>();
    auto *context =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    context->loadAllAvailableDialects();
    return context;
  }();
  return *value;
}

ArtifactRootReference buildDataflow(const ArtifactStore &store) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @sync(%start: none, %value: i32) -> i32
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %result:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%result#1 : i32) streams() memories()
        complete(%result#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %value: i32) ctrl (%ctrl: none) {
    %result, %done = dataflow.graph.launch @sync deps(%ctrl)
        values(%value) stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host() {
    %value = arith.constant 7 : i32
    %thread = dataflow.thread.launch @worker(%value)
        : (i32) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context());
  require(static_cast<bool>(module), "cannot parse the Dataflow fixture");
  auto dataflow = take(dataflow::finalizeCanonicalDataflow(*module));
  return take(dataflow::publishCanonicalDataflow(dataflow, store));
}

loom::fabric::FinalizedFabricRoot buildFabric(const ArtifactStore &store) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  fabric.module @root(%lhs: !fabric.bits<32>, %rhs: !fabric.bits<32>) {
    %result = fabric.pe [spatial]
      (%pe_lhs = %lhs : !fabric.bits<32>,
       %pe_rhs = %rhs : !fabric.bits<32>) -> (!fabric.bits<32>) {
      fabric.fu(%fu_lhs = %pe_lhs : !fabric.bits<32>,
                %fu_rhs = %pe_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
        %sum = fabric.op [@arith.addi] (%fu_lhs, %fu_rhs)
          {implementation_family =
             #fabric.implementation_family<ScalarIntegerAddSub>,
           hw_params = {integer_widths = [32 : i32]}}
          : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %sum : !fabric.bits<32>
      }
    }
    fabric.yield
  }
}
)mlir",
                                              &context());
  require(static_cast<bool>(module), "cannot parse the Fabric fixture");
  const std::vector<std::uint8_t> encoded =
      take(::fabric::encodeResourceContractRecord(
          ::fabric::oneCycleElasticOperationResourceContract()));
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(encoded.size());
  for (std::uint8_t byte : encoded)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  module->walk([&](::fabric::OpOp operation) {
    operation->setAttr(::fabric::kResourceContractRecordAttrName,
                       mlir::DenseI8ArrayAttr::get(&context(), signedBytes));
  });
  ::fabric::ModuleOp root;
  for (::fabric::ModuleOp candidate : module->getOps<::fabric::ModuleOp>())
    root = candidate;
  require(static_cast<bool>(root), "fixture has no Fabric root");
  return take(loom::fabric::finalizeFabricRoot(root, store));
}

SubjectTargetRef rootTarget(const ArtifactRootReference &fabric) {
  return {CaseSubjectRoleRef(0), fabric, SubjectTarget{fabric}};
}

EvaluationCase makeCase(const ArtifactRootReference &fabric,
                        llvm::ArrayRef<EvaluationCondition> conditions,
                        const CaseArtifactResolution &resolution,
                        const ArtifactStore &artifacts,
                        const BlobStore &blobs) {
  EvaluationSubjectBindings subjects =
      take(EvaluationSubjectBindings::get({{CaseSubjectRoleRef(0), {fabric}}}));
  return take(EvaluationCase::get(
      fabricHardwareAnalysisCaseSignatureRef(), std::move(subjects),
      std::nullopt, std::nullopt, conditions, resolution, artifacts, blobs));
}

EvaluationEvidence evaluate(const EvaluationCase &evaluationCase,
                            llvm::ArrayRef<MetricKind> metrics,
                            const CaseArtifactResolution &resolution,
                            const ArtifactStore &artifacts,
                            const BlobStore &blobs) {
  std::vector<MetricRequest> requests;
  requests.reserve(metrics.size());
  for (MetricKind metric : metrics)
    requests.push_back(
        take(MetricRequest::get({metric, EvaluationScope{ScopeFormRef(0), {}}},
                                {}, evaluationCase, resolution, artifacts)));
  ResolvedModelBinding model = take(ResolvedModelBinding::project(
      models::fabricLowConfidenceModelDescriptorRef(), {},
      defaultResolvedConfig()));
  EvaluationRequest request = take(
      EvaluationRequest::get(evaluationCase, requests, {}, std::move(model), 0,
                             resolution, artifacts, blobs));
  take(publishEvaluationRequest(request, artifacts));
  return take(evaluateRequest(request, resolution, artifacts, blobs));
}

DecimalValue pointValue(const EvaluationEvidence &evidence) {
  const auto *completed = std::get_if<CompletedEvidence>(&evidence.outcome());
  require(completed && completed->metricResults.size() == 1,
          "expected exactly one completed metric");
  const auto *point = std::get_if<PointObservation>(
      &completed->metricResults.front().observation);
  const auto *value =
      point ? std::get_if<DecimalValue>(&point->value) : nullptr;
  require(value, "completed metric is not a decimal point");
  return *value;
}

void providerHonorsActivityAndPublishesPhysicalMetrics() {
  TemporaryDirectory directory;
  ArtifactStore artifacts(directory.path());
  BlobStore blobs(directory.path());
  requireSuccess(registerProductionEvaluationRegistry());
  const loom::fabric::FinalizedFabricRoot fabric = buildFabric(artifacts);
  const CaseArtifactResolution resolution =
      take(CaseArtifactResolution::get({{fabric.reference(), {}}}));

  const EvaluationCase staticCase =
      makeCase(fabric.reference(), {}, resolution, artifacts, blobs);
  const std::array staticMetrics = {MetricKind::LimitingClockFrequency,
                                    MetricKind::TotalArea,
                                    MetricKind::LeakagePower};
  const EvaluationEvidence physical =
      evaluate(staticCase, staticMetrics, resolution, artifacts, blobs);
  const auto *completed = std::get_if<CompletedEvidence>(&physical.outcome());
  require(completed && completed->metricResults.size() == staticMetrics.size(),
          "static hardware evaluation did not complete all FPA metrics");
  for (const MetricResult &result : completed->metricResults) {
    const auto *point = std::get_if<PointObservation>(&result.observation);
    const auto *value =
        point ? std::get_if<DecimalValue>(&point->value) : nullptr;
    require(value && value->coefficient() > 0,
            "static hardware evaluation returned a nonpositive metric");
  }

  const EvaluationEvidence missingActivity = evaluate(
      staticCase, {MetricKind::DynamicPower}, resolution, artifacts, blobs);
  require(missingActivity.outcomeKind() == EvidenceOutcomeKind::Unsupported,
          "dynamic power without activity did not remain Unsupported");

  const SubjectTargetRef target = rootTarget(fabric.reference());
  const std::vector<EvaluationCondition> activity = {
      EvaluationCondition{ActivityBindingCondition{
          target, ExplicitAssumptionSource{target, take(ExactRatio::get(1, 2)),
                                           take(ExactRatio::get(1, 10))}}}};
  const EvaluationCase dynamicCase =
      makeCase(fabric.reference(), activity, resolution, artifacts, blobs);
  const DecimalValue dynamicValue = pointValue(evaluate(
      dynamicCase, {MetricKind::DynamicPower}, resolution, artifacts, blobs));
  require(dynamicValue.coefficient() > 0,
          "explicit activity produced zero dynamic power");

  const std::vector<EvaluationCondition> higherActivity = {
      EvaluationCondition{ActivityBindingCondition{
          target, ExplicitAssumptionSource{target, take(ExactRatio::get(1, 1)),
                                           take(ExactRatio::get(1, 2))}}}};
  const EvaluationCase higherActivityCase = makeCase(
      fabric.reference(), higherActivity, resolution, artifacts, blobs);
  const DecimalValue higherDynamic =
      pointValue(evaluate(higherActivityCase, {MetricKind::DynamicPower},
                          resolution, artifacts, blobs));
  require(compareDecimalValue(higherDynamic, dynamicValue) > 0,
          "higher explicit activity did not increase dynamic power");
}

void systemAttachmentMultiplicityIsCounted() {
  TemporaryDirectory directory;
  ArtifactStore artifacts(directory.path());
  BlobStore blobs(directory.path());
  requireSuccess(registerProductionEvaluationRegistry());
  auto design = take(loom::adg::buildBuiltinTarget(
      artifacts, loom::adg::BuiltinTargetPreset::Small));
  require(design.roots().size() == 1,
          "builtin fixture did not publish one System root");
  const loom::fabric::FinalizedFabricRoot &system = design.roots().front();
  require(system.directDependencies().size() == 1,
          "builtin System did not retain one imported SpatialCore");
  const loom::fabric::FinalizedFabricRoot module =
      take(loom::fabric::importEntireFabricRoot(
          system.directDependencies().front().root, artifacts));

  const CaseArtifactResolution moduleResolution =
      take(CaseArtifactResolution::get({{module.reference(), {}}}));
  const CaseArtifactResolution systemResolution =
      take(CaseArtifactResolution::get(
          {{module.reference(), {}},
           {system.reference(), {module.reference()}}}));
  const DecimalValue moduleArea = pointValue(evaluate(
      makeCase(module.reference(), {}, moduleResolution, artifacts, blobs),
      {MetricKind::TotalArea}, moduleResolution, artifacts, blobs));
  const DecimalValue systemArea = pointValue(evaluate(
      makeCase(system.reference(), {}, systemResolution, artifacts, blobs),
      {MetricKind::TotalArea}, systemResolution, artifacts, blobs));
  require(compareDecimalValue(systemArea, moduleArea) > 0,
          "System attachment multiplicity did not increase total area");
}

void frozenWeightDrivesInProcessFpaAndPreservesOod() {
  TemporaryDirectory directory;
  ArtifactStore artifacts(directory.path());
  BlobStore blobs(directory.path());
  requireSuccess(registerProductionEvaluationRegistry());
  const loom::fabric::FinalizedFabricRoot fabric = buildFabric(artifacts);
  const ArtifactRootReference dataflow = buildDataflow(artifacts);
  const ResolvedConfig config = defaultResolvedConfig();

  auto analytic = take(models::prepareCanonicalDataflowFabricEvaluation(
      dataflow, fabric.reference(), config, artifacts, blobs));
  const EvaluationModelDescriptor *descriptor =
      analytic.request.modelBinding().descriptorRef().descriptor();
  require(descriptor, "analytic request lost its model descriptor");
  EvaluationCase evaluationCase = take(EvaluationCase::get(
      descriptor->caseSignature, analytic.request.subjectBindings(),
      analytic.request.workload(), analytic.request.runtimeInput(),
      analytic.request.baseConditions(), analytic.resolution, artifacts,
      blobs));
  OwnerValue projected = take(projectModelFeatures(
      models::fpaModelParameterContractRef(), evaluationCase,
      analytic.resolution, artifacts, blobs));
  const auto *features = projected.getIf<models::FpaFeatureView>();
  require(features, "FPA contract returned a foreign feature view");

  const models::FpaMetricPredictionView observation{
      take(DecimalValue::get(5, 8)), take(DecimalValue::get(2, -6)),
      take(DecimalValue::get(3, -3)), take(DecimalValue::get(4, -4))};
  models::FpaGbdtParameters parameters = take(models::trainFpaGbdtParameters(
      {{*features, observation, {0x21}, {0x31}}},
      models::FpaGbdtTrainingConfig{7, 1, 1, 1, 1, 1}));
  auto bundle = take(finalizeModelParameterBundle(
      models::fpaModelParameterContractRef(),
      OwnerValue::get(std::move(parameters)), artifacts, blobs));
  models::EdaPredictionModelWeight weight =
      take(models::importEdaPredictionModelWeight(bundle.reference(), artifacts,
                                                  blobs));

  models::CanonicalDataflowFabricFpaInference directInference =
      take(models::inferCanonicalDataflowFabricFpa(
          dataflow, fabric.reference(), weight, {}, artifacts, blobs));
  require(
      std::holds_alternative<ModelParameterPrediction>(directInference.outcome),
      "resource-time FPA inference did not consume the frozen weight");

  auto calibrated =
      take(models::prepareCanonicalDataflowFabricCalibratedFpaEvaluation(
          dataflow, fabric.reference(), weight, {}, config, artifacts, blobs));
  require(calibrated.request.modelBinding().findInputBinding(
              ModelInputSlotRef(0)) &&
              calibrated.request.modelBinding()
                      .findInputBinding(ModelInputSlotRef(0))
                      ->artifacts ==
                  std::vector<ArtifactRootReference>{weight.reference()},
          "calibrated request lost the immutable weight binding");
  EvaluationEvidence completed = take(evaluateRequest(
      calibrated.request, calibrated.resolution, artifacts, blobs));
  const auto *completedOutcome =
      std::get_if<CompletedEvidence>(&completed.outcome());
  require(completedOutcome && completedOutcome->metricResults.size() == 4,
          "calibrated FPA did not complete all physical metrics");
  for (std::size_t ordinal = 0; ordinal != 4; ++ordinal) {
    const auto *point = std::get_if<PointObservation>(
        &completedOutcome->metricResults[ordinal].observation);
    const auto *value =
        point ? std::get_if<DecimalValue>(&point->value) : nullptr;
    const std::array expected = {
        observation.limitingClockFrequency, observation.totalArea,
        observation.dynamicPower, observation.leakagePower};
    require(value && *value == expected[ordinal],
            "calibrated FPA changed a frozen model prediction");
  }

  const SubjectTargetRef target{
      models::canonicalDataflowFabricAnalyticFabricRole(), fabric.reference(),
      SubjectTarget{fabric.reference()}};
  const std::vector<EvaluationCondition> outOfDomain = {
      EvaluationCondition{ActivityBindingCondition{
          target, ExplicitAssumptionSource{target, take(ExactRatio::get(1, 2)),
                                           take(ExactRatio::get(1, 10))}}}};
  models::CanonicalDataflowFabricFpaInference directOutOfDomain =
      take(models::inferCanonicalDataflowFabricFpa(
          dataflow, fabric.reference(), weight, outOfDomain, artifacts, blobs));
  require(std::holds_alternative<OutOfDomainModelParameterInference>(
              directOutOfDomain.outcome) &&
              directOutOfDomain.contextDigest != directInference.contextDigest,
          "resource-time FPA inference lost typed OOD or condition context");
  auto oodRequest =
      take(models::prepareCanonicalDataflowFabricCalibratedFpaEvaluation(
          dataflow, fabric.reference(), weight, outOfDomain, config, artifacts,
          blobs));
  EvaluationEvidence unsupported = take(evaluateRequest(
      oodRequest.request, oodRequest.resolution, artifacts, blobs));
  require(unsupported.outcomeKind() == EvidenceOutcomeKind::Unsupported,
          "out-of-domain calibrated FPA did not remain Unsupported");
}

} // namespace

int main() {
  providerHonorsActivityAndPublishesPhysicalMetrics();
  systemAttachmentMultiplicityIsCounted();
  frozenWeightDrivesInProcessFpaAndPreservesOod();
  return EXIT_SUCCESS;
}
