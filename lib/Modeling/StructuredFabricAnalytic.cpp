#include "Evaluation/Models/StructuredFabricAnalytic.h"
#include "Evaluation/ProductionRegistry.h"

#include "AnalyticModelSupport.h"
#include "StructuredEvaluationInvocationCacheInternal.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/MappingDebugLog.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Evaluation/ModelProvider.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/IR/LoomOps.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"
#include "Frontend/Lowering/GraphParallelLowering.h"
#include "Simulator/NativeSimulationOracle.h"
#include "Simulator/SimulationArtifacts.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace loom::evaluation::models {
namespace {

constexpr BuiltinEvaluationCase kCase =
    BuiltinEvaluationCase::StructuredProgramWithFabric;
constexpr BuiltinEvaluationModel kModel =
    BuiltinEvaluationModel::StructuredFabricLowConfidence;
constexpr CaseSubjectRoleRef kStructuredProgramRole(0);
constexpr CaseSubjectRoleRef kFabricRole(1);

EvaluationCaseSignatureRef caseSignatureRef() {
  return llvm::cantFail(EvaluationCaseSignatureRef::get(
      evaluationSchemaVersion(), builtinEvaluationCaseKind(kCase)));
}

const ArtifactSchemaDescriptor *const kStructuredSchemas[] = {
    &frontend::structuredProgramArtifactSchema};
const ArtifactSchemaDescriptor *const kFabricSchemas[] = {
    &fabric::fabricArtifactSchema};
const ArtifactSchemaDescriptor *const kWorkloadSchemas[] = {
    &sim::simulationWorkloadSchema};
const ArtifactSchemaDescriptor *const kRuntimeInputSchemas[] = {
    &sim::simulationRuntimeInputSchema};

const CaseSubjectRoleDescriptor kSubjectRoles[] = {
    {kStructuredProgramRole, "structured_program",
     SubjectRoleCardinality::ExactlyOne, kStructuredSchemas, nullptr},
    {kFabricRole, "fabric", SubjectRoleCardinality::ExactlyOne, kFabricSchemas,
     nullptr},
};

SubjectTargetPattern fabricRootPattern() {
  return SubjectTargetPattern{
      kFabricRole,
      SubjectReferenceType{ArtifactRootType{fabric::fabricArtifactSchema}}};
}

const std::vector<ConditionApplicabilityPattern> kBaseConditionPatterns = {
    {EvaluationConditionKind::ProcessCorner,
     {caseSignatureRef(), {fabricRootPattern()}}},
    {EvaluationConditionKind::SupplyVoltage,
     {caseSignatureRef(), {fabricRootPattern()}}},
    {EvaluationConditionKind::Temperature,
     {caseSignatureRef(), {fabricRootPattern()}}},
    {EvaluationConditionKind::RequiredClockPeriod,
     {caseSignatureRef(), {fabricRootPattern()}}},
    {EvaluationConditionKind::RelativeClockSchedule,
     {caseSignatureRef(), {fabricRootPattern(), fabricRootPattern()}}},
    {EvaluationConditionKind::ActivityBinding,
     {caseSignatureRef(), {fabricRootPattern()}}},
    {EvaluationConditionKind::ActivityBinding,
     {caseSignatureRef(), {fabricRootPattern(), fabricRootPattern()}}},
};

llvm::Error verifyWorkloadCompatibility(
    const EvaluationCase &, const EvaluationSubjectBindings &,
    const std::optional<ArtifactRootReference> &workload,
    const std::optional<ArtifactRootReference> &runtimeInput,
    const CaseArtifactResolution &resolution, const ArtifactStore &,
    const BlobStore &) {
  if (!workload || !runtimeInput)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_invalid: workload inputs are not total");
  const CaseArtifactResolution::Entry *workloadEntry =
      resolution.find(*workload);
  const CaseArtifactResolution::Entry *runtimeEntry =
      resolution.find(*runtimeInput);
  if (!workloadEntry || !runtimeEntry ||
      !CaseArtifactResolution::reaches(*runtimeEntry, *workload))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_invalid: runtime input does not reach its "
        "exact workload");
  bool hasStructuredOwner = false;
  for (const ArtifactRootReference &dependency :
       workloadEntry->dependencyClosure)
    hasStructuredOwner |=
        dependency.schemaIdentity ==
            frontend::structuredProgramArtifactSchema.identity &&
        dependency.schemaVersion ==
            frontend::structuredProgramArtifactSchema.version;
  if (!hasStructuredOwner)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_invalid: workload has no exact Structured "
        "Program owner");
  return llvm::Error::success();
}

const EvaluationCaseSignatureDescriptor kCaseSignature{
    builtinEvaluationCaseKind(kCase),
    "structured_program_with_fabric",
    "One exact Structured Program evaluated against one exact Fabric.",
    kSubjectRoles,
    ArtifactRequirement::Required,
    kWorkloadSchemas,
    ArtifactRequirement::Required,
    kRuntimeInputSchemas,
    &verifyWorkloadCompatibility,
    AbsentReferenceCycle{},
    kBaseConditionPatterns};

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
    builtinEvaluationModelKind(kModel),
    "structured_fabric_low_confidence",
    "loom.structured_fabric.low_confidence.v4",
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
    {},
    ProviderForm::InProcess};

bool isInsideGlobal(mlir::Operation *operation) {
  return static_cast<bool>(operation->getParentOfType<mlir::LLVM::GlobalOp>());
}

bool isExecutableLeaf(mlir::Operation *operation) {
  if (!sim::isNativeStructuredProfileBlock(operation->getBlock()) ||
      operation->getNumRegions() != 0 ||
      operation->hasTrait<mlir::OpTrait::IsTerminator>() ||
      mlir::isa<mlir::SymbolOpInterface>(operation) ||
      isInsideGlobal(operation))
    return false;
  return true;
}

struct BlockActivityProjection final {
  frontend::StructuredProgramCandidateView view;
  llvm::DenseMap<mlir::Block *, std::uint64_t> activations;
  std::uint64_t hostInstructionLeafExecutions = 0;
};

struct ScopeDynamicWork final {
  std::uint64_t instructionLeafExecutions = 0;
  std::uint64_t dynamicActivations = 0;
};

struct SpatialDynamicWork final {
  std::uint64_t dynamicLeafExecutions = 0;
  std::uint64_t loweredLeafCopies = 0;
};

struct ResolvedScopeActivity final {
  frontend::StructuredEntity selected;
  std::uint64_t dynamicActivations = 0;
};

using CachedMetrics = std::optional<detail::LowConfidenceMetricSet>;

detail::StructuredAnalyticCacheKey
metricCacheKey(const ArtifactRootReference &structuredProgram,
               const ArtifactRootReference &fabricReference,
               const ArtifactRootReference &workload,
               const ArtifactRootReference &runtimeInput,
               const ComponentViewDigest &configDigest) {
  return {structuredProgram, fabricReference, workload, runtimeInput,
          configDigest};
}

bool isInsideSpatialRegion(mlir::Operation *operation) {
  return operation &&
         (llvm::isa<::loom::SpatialRegionOp>(operation) ||
          static_cast<bool>(
              operation->getParentOfType<::loom::SpatialRegionOp>()));
}

bool isModeledHostLeaf(mlir::Operation *operation) {
  return isExecutableLeaf(operation) && !isInsideSpatialRegion(operation) &&
         !llvm::isa<dataflow::ThreadLaunchOp, dataflow::ThreadWaitOp>(
             operation);
}

llvm::Expected<std::uint64_t> checkedScaledCount(std::uint64_t count,
                                                 std::uint64_t activations,
                                                 llvm::StringRef context) {
  const std::optional<std::uint64_t> result =
      llvm::checkedMulUnsigned(count, activations);
  if (!result)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "structured_fabric_model_overflow: %s",
                                   context.str().c_str());
  return *result;
}

llvm::Expected<std::uint64_t>
fixedParallelCardinality(mlir::Operation *operation) {
  std::optional<lowering::FixedParallelDomain> domain =
      lowering::getFixedParallelDomain(operation);
  if (!domain || domain->lower.size() != domain->upper.size() ||
      domain->lower.size() != domain->step.size())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_invalid: graph-owned parallel operation "
        "has no fixed domain");

  std::uint64_t cardinality = 1;
  for (auto [lower, upper, step] :
       llvm::zip_equal(domain->lower, domain->upper, domain->step)) {
    if (step <= 0)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_fabric_model_invalid: graph-owned parallel operation "
          "has a non-positive step");
    if (upper <= lower)
      return std::uint64_t{0};
    const unsigned __int128 extent = static_cast<unsigned __int128>(
        static_cast<__int128>(upper) - static_cast<__int128>(lower));
    const unsigned __int128 dimension =
        (extent + static_cast<unsigned __int128>(step) - 1) /
        static_cast<unsigned __int128>(step);
    const unsigned __int128 product =
        static_cast<unsigned __int128>(cardinality) * dimension;
    if (product > std::numeric_limits<std::uint64_t>::max())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_fabric_model_overflow: fixed parallel lane count");
    cardinality = static_cast<std::uint64_t>(product);
  }
  return cardinality;
}

llvm::Expected<SpatialDynamicWork>
projectSpatialDynamicWork(const BlockActivityProjection &activity,
                          ::loom::SpatialRegionOp spatial) {
  SpatialDynamicWork result;
  llvm::DenseMap<mlir::Operation *, std::uint64_t> parallelCardinalities;
  llvm::Error failure = llvm::Error::success();
  spatial.walk([&](mlir::Operation *operation) {
    if (failure || !isExecutableLeaf(operation))
      return;
    auto activation = activity.activations.find(operation->getBlock());
    if (activation == activity.activations.end()) {
      failure = llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_fabric_model_invalid: Spatial leaf has no dynamic "
          "activity projection");
      return;
    }
    const std::optional<std::uint64_t> dynamic = llvm::checkedAddUnsigned(
        result.dynamicLeafExecutions, activation->second);
    if (!dynamic) {
      failure = llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_fabric_model_overflow: Spatial dynamic leaf work");
      return;
    }
    result.dynamicLeafExecutions = *dynamic;

    std::uint64_t copies = 1;
    for (mlir::Operation *parent = operation->getParentOp();
         parent && parent != spatial.getOperation();
         parent = parent->getParentOp()) {
      if (!llvm::isa<mlir::scf::ForallOp, mlir::scf::ParallelOp>(parent))
        continue;
      auto found = parallelCardinalities.find(parent);
      if (found == parallelCardinalities.end()) {
        auto cardinality = fixedParallelCardinality(parent);
        if (!cardinality) {
          failure = cardinality.takeError();
          return;
        }
        found = parallelCardinalities.try_emplace(parent, *cardinality).first;
      }
      const std::optional<std::uint64_t> product =
          llvm::checkedMulUnsigned(copies, found->second);
      if (!product) {
        failure = llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "structured_fabric_model_overflow: lowered parallel leaf copies");
        return;
      }
      copies = *product;
    }
    const std::optional<std::uint64_t> totalCopies =
        llvm::checkedAddUnsigned(result.loweredLeafCopies, copies);
    if (!totalCopies) {
      failure = llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_fabric_model_overflow: lowered Spatial leaf copies");
      return;
    }
    result.loweredLeafCopies = *totalCopies;
  });
  if (failure)
    return std::move(failure);

  if (result.loweredLeafCopies == 0) {
    auto activation = activity.activations.find(&spatial.getBody().front());
    if (activation == activity.activations.end())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_fabric_model_invalid: empty Spatial owner has no "
          "activity projection");
    result.dynamicLeafExecutions = activation->second;
    result.loweredLeafCopies = 1;
  }
  return result;
}

llvm::Expected<BlockActivityProjection> projectBlockActivity(
    const frontend::StructuredProgramCandidate &source,
    const sim::NativeStructuredProgramObservations &observations) {
  llvm::Expected<frontend::StructuredProgramCandidateView> view = source.view();
  if (!view)
    return view.takeError();

  std::vector<const frontend::StructuredEntity *> expectedBlocks;
  for (const frontend::StructuredEntity &entity :
       view->entities(frontend::StructuredEntityKind::Block))
    if (sim::isNativeStructuredProfileBlock(entity.block))
      expectedBlocks.push_back(&entity);
  if (expectedBlocks.size() != observations.blockActivations.size())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_invalid: block activation projection is not "
        "total");

  BlockActivityProjection projection{
      std::move(*view), llvm::DenseMap<mlir::Block *, std::uint64_t>(), 0};
  for (std::size_t ordinal = 0; ordinal < expectedBlocks.size(); ++ordinal) {
    const sim::NativeStructuredBlockActivation &observed =
        observations.blockActivations[ordinal];
    if (observed.block != expectedBlocks[ordinal]->reference)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_fabric_model_invalid: block activation order differs "
          "from the exact Structured owner");
    mlir::Block *block = expectedBlocks[ordinal]->block;
    projection.activations[block] = observed.activations;
    const std::uint64_t leaves =
        llvm::count_if(*block, [](mlir::Operation &operation) {
          return isModeledHostLeaf(&operation);
        });
    llvm::Expected<std::uint64_t> dynamicLeaves = checkedScaledCount(
        leaves, observed.activations, "source instruction executions");
    if (!dynamicLeaves)
      return dynamicLeaves.takeError();
    const std::optional<std::uint64_t> updated = llvm::checkedAddUnsigned(
        projection.hostInstructionLeafExecutions, *dynamicLeaves);
    if (!updated)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_fabric_model_overflow: source instruction total");
    projection.hostInstructionLeafExecutions = *updated;
  }
  return projection;
}

llvm::Expected<BlockActivityProjection> projectBlockActivity(
    const frontend::StructuredProgramCandidate &candidate,
    const BlockActivityProjection &source,
    llvm::ArrayRef<frontend::StructuredBlockActivityLineage> lineage) {
  auto view = candidate.view();
  if (!view)
    return view.takeError();
  BlockActivityProjection projection{
      std::move(*view), llvm::DenseMap<mlir::Block *, std::uint64_t>(), 0};
  llvm::DenseSet<std::uint64_t> seenChildren;
  for (const frontend::StructuredBlockActivityLineage &entry : lineage) {
    if (entry.childBlock.parent != projection.view.identity() ||
        entry.childBlock.kind != frontend::StructuredEntityKind::Block ||
        entry.parentBlock.parent != source.view.identity() ||
        entry.parentBlock.kind != frontend::StructuredEntityKind::Block)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_fabric_model_invalid: block activity lineage has the "
          "wrong owner or kind");
    if (!seenChildren.insert(entry.childBlock.ordinal).second)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_fabric_model_invalid: block activity lineage is not "
          "injective");
    auto child = projection.view.resolve(entry.childBlock);
    if (!child)
      return child.takeError();
    auto parent = source.view.resolve(entry.parentBlock);
    if (!parent)
      return parent.takeError();
    if (!child->block || !parent->block)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_fabric_model_invalid: block activity lineage did not "
          "resolve to blocks");
    if (!sim::isNativeStructuredProfileBlock(child->block))
      continue;
    auto activation = source.activations.find(parent->block);
    if (activation == source.activations.end())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_fabric_model_invalid: parent block has no dynamic "
          "activity projection");
    projection.activations.try_emplace(child->block, activation->second);
  }

  for (const frontend::StructuredEntity &entity :
       projection.view.entities(frontend::StructuredEntityKind::Block)) {
    if (!sim::isNativeStructuredProfileBlock(entity.block))
      continue;
    auto activation = projection.activations.find(entity.block);
    if (activation == projection.activations.end())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_fabric_model_invalid: block activity lineage is not "
          "total at canonical block %llu owned by %s",
          static_cast<unsigned long long>(entity.reference.ordinal),
          entity.block->getParentOp()->getName().getStringRef().str().c_str());
    const std::uint64_t leaves =
        llvm::count_if(*entity.block, [](mlir::Operation &operation) {
          return isModeledHostLeaf(&operation);
        });
    auto dynamicLeaves = checkedScaledCount(leaves, activation->second,
                                            "candidate instruction executions");
    if (!dynamicLeaves)
      return dynamicLeaves.takeError();
    const std::optional<std::uint64_t> updated = llvm::checkedAddUnsigned(
        projection.hostInstructionLeafExecutions, *dynamicLeaves);
    if (!updated)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_fabric_model_overflow: candidate instruction total");
    projection.hostInstructionLeafExecutions = *updated;
  }
  return projection;
}

llvm::Expected<ResolvedScopeActivity>
resolveScopeActivity(const BlockActivityProjection &projection,
                     const frontend::StructuredEntityRef &sourceScope) {
  llvm::Expected<frontend::StructuredEntity> selected =
      projection.view.resolve(sourceScope);
  if (!selected)
    return selected.takeError();
  if (!selected->operation)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_invalid: source scope is not an operation");

  mlir::Block *activationBlock = nullptr;
  if (auto function =
          llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(selected->operation))
    activationBlock =
        function.getBody().empty() ? nullptr : &function.getBody().front();
  else
    activationBlock = selected->operation->getBlock();
  auto foundActivation = projection.activations.find(activationBlock);
  if (!activationBlock || foundActivation == projection.activations.end())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_invalid: source scope has no activation "
        "projection");
  return ResolvedScopeActivity{std::move(*selected), foundActivation->second};
}

llvm::Expected<ScopeDynamicWork>
projectScopeDynamicWork(const BlockActivityProjection &projection,
                        const frontend::StructuredEntityRef &sourceScope) {
  auto activity = resolveScopeActivity(projection, sourceScope);
  if (!activity)
    return activity.takeError();

  std::uint64_t coveredLeaves = 0;
  llvm::Error failure = llvm::Error::success();
  activity->selected.operation->walk([&](mlir::Operation *operation) {
    if (failure)
      return mlir::WalkResult::interrupt();
    if (operation != activity->selected.operation &&
        llvm::isa<mlir::FunctionOpInterface>(operation))
      return mlir::WalkResult::skip();
    if (!isExecutableLeaf(operation))
      return mlir::WalkResult::advance();
    auto found = projection.activations.find(operation->getBlock());
    if (found == projection.activations.end()) {
      failure = llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_fabric_model_invalid: selected leaf has no block "
          "activation");
      return mlir::WalkResult::interrupt();
    }
    const std::optional<std::uint64_t> updated =
        llvm::checkedAddUnsigned(coveredLeaves, found->second);
    if (!updated) {
      failure = llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_fabric_model_overflow: selected instruction total");
      return mlir::WalkResult::interrupt();
    }
    coveredLeaves = *updated;
    return mlir::WalkResult::advance();
  });
  if (failure)
    return std::move(failure);
  if (coveredLeaves > projection.hostInstructionLeafExecutions)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_invalid: selected work exceeds source work");
  return ScopeDynamicWork{coveredLeaves, activity->dynamicActivations};
}

llvm::Error accumulateScaledCount(std::uint64_t &destination,
                                  std::uint64_t value,
                                  std::uint64_t activations,
                                  llvm::StringRef context) {
  auto scaled = checkedScaledCount(value, activations, context);
  if (!scaled)
    return scaled.takeError();
  const std::optional<std::uint64_t> updated =
      llvm::checkedAddUnsigned(destination, *scaled);
  if (!updated)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_overflow: accumulated %s",
        context.str().c_str());
  destination = *updated;
  return llvm::Error::success();
}

llvm::Error accumulateScaledRatio(std::uint64_t &destination,
                                  std::uint64_t value, std::uint64_t numerator,
                                  std::uint64_t denominator,
                                  llvm::StringRef context) {
  if (denominator == 0)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_invalid: zero %s denominator",
        context.str().c_str());
  const unsigned __int128 product =
      static_cast<unsigned __int128>(value) * numerator;
  const unsigned __int128 scaled =
      product / denominator + (product % denominator != 0 ? 1 : 0);
  if (scaled > std::numeric_limits<std::uint64_t>::max())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_overflow: scaled %s", context.str().c_str());
  const std::optional<std::uint64_t> updated =
      llvm::checkedAddUnsigned(destination, static_cast<std::uint64_t>(scaled));
  if (!updated)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_overflow: accumulated %s",
        context.str().c_str());
  destination = *updated;
  return llvm::Error::success();
}

llvm::Error
accumulateGraphWorkload(detail::AnalyticWorkloadEstimate &destination,
                        const detail::AnalyticWorkloadEstimate &graph,
                        std::uint64_t activations,
                        const SpatialDynamicWork &dynamicWork) {
  if (llvm::Error error = accumulateScaledRatio(
          destination.schedulingPressure, graph.schedulingPressure,
          dynamicWork.dynamicLeafExecutions, dynamicWork.loweredLeafCopies,
          "Spatial scheduling pressure"))
    return error;
  if (llvm::Error error = accumulateScaledRatio(
          destination.activityUnits, graph.activityUnits,
          dynamicWork.dynamicLeafExecutions, dynamicWork.loweredLeafCopies,
          "Spatial activity"))
    return error;
  if (llvm::Error error = accumulateScaledCount(
          destination.graphActivations, graph.graphActivations, activations,
          "graph activations"))
    return error;
  if (llvm::Error error = accumulateScaledCount(
          destination.boundaryPayloadBytes, graph.boundaryPayloadBytes,
          activations, "boundary payload bytes"))
    return error;
  if (llvm::Error error = accumulateScaledCount(
          destination.memoryBoundaryBindings, graph.memoryBoundaryBindings,
          activations, "memory boundary bindings"))
    return error;
  return accumulateScaledRatio(
      destination.memoryTransactions, graph.memoryTransactions,
      dynamicWork.dynamicLeafExecutions, dynamicWork.loweredLeafCopies,
      "memory transactions");
}

llvm::Expected<std::optional<detail::LowConfidenceMetricSet>> estimateMetrics(
    const BlockActivityProjection &activity,
    const fabric::FinalizedFabricRoot &fabricRoot,
    const dataflow::CanonicalDataflowProgramView *projectedDataflow,
    llvm::ArrayRef<lowering::StructuredSpatialGraphProjection> spatialGraphs) {
  std::size_t spatialRegionCount = 0;
  for (const frontend::StructuredEntity &entity :
       activity.view.entities(frontend::StructuredEntityKind::Operation))
    spatialRegionCount +=
        llvm::isa_and_nonnull<::loom::SpatialRegionOp>(entity.operation);
  if (spatialRegionCount != spatialGraphs.size())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_invalid: Spatial graph projection is not "
        "total");
  if ((projectedDataflow != nullptr) != !spatialGraphs.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_invalid: Dataflow projection is partial");

  detail::AnalyticWorkloadEstimate pressure;
  llvm::DenseSet<mlir::Operation *> visitedRegions;
  llvm::DenseSet<mlir::Operation *> visitedLaunches;
  for (const lowering::StructuredSpatialGraphProjection &projection :
       spatialGraphs) {
    auto region = activity.view.resolve(projection.spatialRegion);
    if (!region)
      return region.takeError();
    auto spatial =
        llvm::dyn_cast_or_null<::loom::SpatialRegionOp>(region->operation);
    if (!spatial || !spatial.getBody().hasOneBlock())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_fabric_model_invalid: projected Spatial owner has the "
          "wrong shape");
    if (!visitedRegions.insert(region->operation).second)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_fabric_model_invalid: Spatial graph projection is not "
          "injective");
    auto found = activity.activations.find(&spatial.getBody().front());
    if (found == activity.activations.end())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_fabric_model_invalid: Spatial owner has no dynamic "
          "activity projection");

    auto launch = projectedDataflow->resolve(projection.staticGraphLaunch);
    if (!launch)
      return launch.takeError();
    if (!visitedLaunches.insert(launch->op).second)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_fabric_model_invalid: Spatial graph projection is not "
          "injective");
    auto graph = detail::projectCanonicalDataflowGraphWorkload(
        *projectedDataflow, launch->callee, fabricRoot);
    if (!graph)
      return graph.takeError();
    if (!*graph)
      return std::optional<detail::LowConfidenceMetricSet>{};
    auto dynamicWork = projectSpatialDynamicWork(activity, spatial);
    if (!dynamicWork)
      return dynamicWork.takeError();
    if (llvm::Error error = accumulateGraphWorkload(
            pressure, **graph, found->second, *dynamicWork))
      return std::move(error);
  }

  auto metrics = detail::estimateLowConfidenceMetrics(
      activity.hostInstructionLeafExecutions, pressure, fabricRoot);
  if (!metrics)
    return metrics.takeError();
  return std::optional<detail::LowConfidenceMetricSet>(std::move(*metrics));
}

llvm::Expected<EvaluationModelResult>
evaluate(const EvaluationRequest &request, const CaseArtifactResolution &,
         const ArtifactStore &artifactStore, const BlobStore &) {
  llvm::ArrayRef<ArtifactRootReference> structured =
      request.subjectBindings().subjects(kStructuredProgramRole);
  llvm::ArrayRef<ArtifactRootReference> fabric =
      request.subjectBindings().subjects(kFabricRole);
  if (structured.size() != 1 || fabric.size() != 1 || !request.workload() ||
      !request.runtimeInput())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_invalid: exact case inputs are not total");

  const detail::StructuredAnalyticCacheKey cacheKey =
      metricCacheKey(structured.front(), fabric.front(), *request.workload(),
                     *request.runtimeInput(),
                     request.modelBinding().resolvedModelConfig().digest());
  StructuredEvaluationInvocationCache *cache =
      detail::currentStructuredEvaluationCache();
  std::shared_ptr<const CachedMetrics> cachedMetrics;
  if (cache) {
    auto &impl = detail::StructuredEvaluationCacheAccess::impl(*cache);
    std::lock_guard<std::mutex> lock(impl.mutex);
    auto found = impl.analyticResults.find(cacheKey);
    if (found != impl.analyticResults.end()) {
      cachedMetrics = found->second;
      impl.analyticHitCount.fetch_add(1, std::memory_order_relaxed);
    } else {
      impl.analyticMissCount.fetch_add(1, std::memory_order_relaxed);
    }
  }

  CachedMetrics metrics;
  if (cachedMetrics) {
    if (auto stored = artifactStore.get(structured.front()); !stored)
      return stored.takeError();
    if (auto stored = artifactStore.get(fabric.front()); !stored)
      return stored.takeError();
    if (auto stored = artifactStore.get(*request.workload()); !stored)
      return stored.takeError();
    if (auto stored = artifactStore.get(*request.runtimeInput()); !stored)
      return stored.takeError();
    metrics = *cachedMetrics;
  } else {
    auto program =
        frontend::importStructuredProgram(structured.front(), artifactStore);
    if (!program)
      return program.takeError();
    auto fabricRoot =
        detail::importCachedFabricRoot(fabric.front(), artifactStore);
    if (!fabricRoot)
      return fabricRoot.takeError();
    auto inputs = sim::importStructuredProgramSimulationInputs(
        *request.workload(), *request.runtimeInput(), artifactStore);
    if (!inputs)
      return inputs.takeError();
    bool hasSpatialOwnership = false;
    program->module().walk(
        [&](::loom::SpatialRegionOp) { hasSpatialOwnership = true; });
    auto observations =
        hasSpatialOwnership ||
                program->identity() != inputs->structuredProgram.identity()
            ? sim::executeProfiledSelectedStructuredProgram(
                  *program, inputs->structuredProgram, inputs->workload,
                  inputs->runtimeInput)
            : sim::executeNativeStructuredProgram(inputs->structuredProgram,
                                                  inputs->workload,
                                                  inputs->runtimeInput);
    if (!observations) {
      std::error_code code;
      std::string message;
      llvm::raw_string_ostream stream(message);
      llvm::handleAllErrors(observations.takeError(),
                            [&](const llvm::ErrorInfoBase &error) {
                              code = error.convertToErrorCode();
                              error.log(stream);
                            });
      stream.flush();
      if (code == std::make_error_code(std::errc::not_supported))
        return EvaluationModelResult{
            {},
            UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
      if (code == std::make_error_code(std::errc::io_error))
        return EvaluationModelResult{
            {}, ExecutionFailedEvidence{OutcomeReason::ToolFailure}};
      return llvm::createStringError(
          code ? code : llvm::inconvertibleErrorCode(), "%s", message.c_str());
    }
    std::optional<lowering::ProjectedCanonicalDataflow> projected;
    std::optional<dataflow::CanonicalDataflowProgramView> projectedView;
    if (hasSpatialOwnership) {
      auto lowered =
          lowering::lowerStructuredProgramToCanonicalDataflowWithProjection(
              *program);
      if (!lowered)
        return lowered.takeError();
      projected.emplace(std::move(*lowered));
      auto view = projected->artifact.view();
      if (!view)
        return view.takeError();
      projectedView.emplace(std::move(*view));
    }
    auto activity = projectBlockActivity(*program, *observations);
    if (!activity)
      return activity.takeError();
    auto computed = estimateMetrics(
        *activity, **fabricRoot, projectedView ? &*projectedView : nullptr,
        projected
            ? llvm::ArrayRef(projected->spatialGraphs)
            : llvm::ArrayRef<lowering::StructuredSpatialGraphProjection>{});
    if (!computed)
      return computed.takeError();
    metrics = std::move(*computed);
    if (cache) {
      auto &impl = detail::StructuredEvaluationCacheAccess::impl(*cache);
      auto cached = std::make_shared<const CachedMetrics>(metrics);
      std::lock_guard<std::mutex> lock(impl.mutex);
      auto [found, inserted] =
          impl.analyticResults.try_emplace(cacheKey, std::move(cached));
      if (!inserted && *found->second != metrics)
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "structured_fabric_model_invalid: nondeterministic cached "
            "result");
      if (inserted)
        impl.analyticPrimeCount.fetch_add(1, std::memory_order_relaxed);
    }
  }
  if (!metrics) {
    mapping_debug::emit(
        mapping_debug::Level::Detail, mapping_debug::Stage::DataflowLowering,
        mapping_debug::Event::MappingFailure, [&](llvm::json::Object &fields) {
          fields["failure_scope"] = "structured_fabric_analytic";
          fields["closure_status"] = "proven_infeasible";
          fields["diagnostic"] =
              "candidate demand has no admitting Fabric resource";
        });
    return EvaluationModelResult{
        {}, UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
  }

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

const EvaluationModelProvider kProvider{
    kModelDescriptor.reference(), EvaluationModelInProcessProvider{&evaluate}};

} // namespace

llvm::Expected<std::vector<StructuredScopeActivityProjection>>
projectStructuredScopeActivity(
    const frontend::StructuredProgramCandidate &sourceProgram,
    const sim::NativeStructuredProgramObservations &sourceObservations,
    llvm::ArrayRef<frontend::StructuredEntityRef> scopes) {
  auto projection = projectBlockActivity(sourceProgram, sourceObservations);
  if (!projection)
    return projection.takeError();
  std::vector<StructuredScopeActivityProjection> result;
  result.reserve(scopes.size());
  for (const frontend::StructuredEntityRef &scope : scopes) {
    auto activity = projectScopeDynamicWork(*projection, scope);
    if (!activity)
      return activity.takeError();
    result.push_back({scope, activity->dynamicActivations,
                      activity->instructionLeafExecutions});
  }
  return result;
}

llvm::Error registerStructuredFabricAnalyticModel() {
  if (llvm::Error error = registerEvaluationCaseSignature(kCaseSignature))
    return error;
  if (llvm::Error error = registerEvaluationModelDescriptor(kModelDescriptor))
    return error;
  return registerEvaluationModelProvider(kProvider);
}

EvaluationModelDescriptorRef structuredFabricAnalyticModelDescriptorRef() {
  return kModelDescriptor.reference();
}

CaseSubjectRoleRef structuredFabricAnalyticCandidateRole() {
  return kStructuredProgramRole;
}

CaseSubjectRoleRef structuredFabricAnalyticFabricRole() { return kFabricRole; }

llvm::Expected<std::int64_t>
structuredFabricAnalyticMetricQuantumBase10Exponent(MetricKind metric) {
  return detail::lowConfidenceMetricQuantumBase10Exponent(metric);
}

llvm::Expected<StructuredFabricAnalyticRequestContext>
prepareStructuredFabricAnalyticInvocation(
    llvm::ArrayRef<ArtifactRootReference> structuredPrograms,
    const ArtifactRootReference &fabricReference,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput,
    const ArtifactStore &artifactStore) {
  if (llvm::Error error = registerStructuredFabricAnalyticModel())
    return std::move(error);
  if (structuredPrograms.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_invalid: analytic invocation has no "
        "Structured candidates");

  std::vector<ArtifactRootReference> candidates;
  candidates.reserve(structuredPrograms.size());
  for (const ArtifactRootReference &root : structuredPrograms) {
    if (root.schemaIdentity !=
            frontend::structuredProgramArtifactSchema.identity ||
        root.schemaVersion != frontend::structuredProgramArtifactSchema.version)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_fabric_model_invalid: analytic invocation contains a "
          "foreign Structured candidate");
    auto stored = artifactStore.get(root);
    if (!stored)
      return stored.takeError();
    candidates.push_back(root);
  }
  llvm::sort(candidates, artifactRootReferenceLess);
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());

  auto inputs = sim::importStructuredProgramSimulationInputs(
      workload, runtimeInput, artifactStore);
  if (!inputs)
    return inputs.takeError();
  const ArtifactRootReference sourceReference{
      frontend::structuredProgramArtifactSchema.identity.str(),
      frontend::structuredProgramArtifactSchema.version,
      inputs->structuredProgram.identity()};

  std::vector<CaseArtifactResolution::Entry> additionalEntries;
  additionalEntries.reserve(candidates.size() + 3);
  additionalEntries.push_back({sourceReference, {}});
  additionalEntries.push_back({workload, {sourceReference}});
  additionalEntries.push_back({runtimeInput, {sourceReference, workload}});
  for (const ArtifactRootReference &candidate : candidates)
    additionalEntries.push_back({candidate, {}});

  auto resolution = detail::resolveSingleSubjectFabricCase(
      candidates.front(), fabricReference, artifactStore, additionalEntries);
  if (!resolution)
    return resolution.takeError();
  return StructuredFabricAnalyticRequestContext(
      std::move(candidates), fabricReference, workload, runtimeInput,
      std::move(*resolution));
}

llvm::Expected<PreparedStructuredFabricEvaluation>
prepareStructuredFabricEvaluation(
    const ArtifactRootReference &structuredProgram,
    const ArtifactRootReference &fabricReference,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ResolvedConfig &config,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  auto invocation = prepareStructuredFabricAnalyticInvocation(
      {structuredProgram}, fabricReference, workload, runtimeInput,
      artifactStore);
  if (!invocation)
    return invocation.takeError();
  return prepareStructuredFabricEvaluation(structuredProgram, *invocation,
                                           config, artifactStore, blobStore);
}

llvm::Expected<PreparedStructuredFabricEvaluation>
prepareStructuredFabricEvaluation(
    const ArtifactRootReference &structuredProgram,
    const StructuredFabricAnalyticRequestContext &invocation,
    const ResolvedConfig &config, const ArtifactStore &artifactStore,
    const BlobStore &blobStore) {
  if (llvm::Error error = registerStructuredFabricAnalyticModel())
    return std::move(error);
  if (!std::binary_search(invocation.candidates_.begin(),
                          invocation.candidates_.end(), structuredProgram,
                          artifactRootReferenceLess))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_invalid: candidate is outside the analytic "
        "invocation");

  auto modelBinding =
      ResolvedModelBinding::project(kModelDescriptor.reference(), {}, config);
  if (!modelBinding)
    return modelBinding.takeError();

  auto bindings = EvaluationSubjectBindings::get(
      {{kStructuredProgramRole, {structuredProgram}},
       {kFabricRole, {invocation.fabric_}}});
  if (!bindings)
    return bindings.takeError();
  auto evaluationCase =
      EvaluationCase::get(caseSignatureRef(), std::move(*bindings),
                          invocation.workload_, invocation.runtimeInput_, {},
                          invocation.resolution_, artifactStore, blobStore);
  if (!evaluationCase)
    return evaluationCase.takeError();
  std::vector<MetricRequest> metrics;
  metrics.reserve(std::size(kMetricCapabilities));
  for (const MetricCapability &capability : kMetricCapabilities) {
    auto metric = MetricRequest::get(
        MetricQuery{capability.kind, EvaluationScope{ScopeFormRef(0), {}}}, {},
        *evaluationCase, invocation.resolution_, artifactStore);
    if (!metric)
      return metric.takeError();
    metrics.push_back(std::move(*metric));
  }
  auto request = EvaluationRequest::get(
      *evaluationCase, metrics, {}, std::move(*modelBinding), 0,
      invocation.resolution_, artifactStore, blobStore);
  if (!request)
    return request.takeError();
  auto published = publishEvaluationRequest(*request, artifactStore);
  if (!published)
    return published.takeError();
  return PreparedStructuredFabricEvaluation{
      std::move(*request), invocation.resolution_, kStructuredProgramRole};
}

llvm::Error primeStructuredFabricAnalyticResult(
    const ArtifactRootReference &structuredProgramReference,
    const StructuredFabricAnalyticCandidateProjection &candidate,
    const StructuredFabricAnalyticInvocation &invocation,
    const fabric::FinalizedFabricRoot &fabricRoot, const ResolvedConfig &config,
    const ArtifactStore &artifactStore) {
  if (llvm::Error error = registerStructuredFabricAnalyticModel())
    return error;
  StructuredEvaluationInvocationCache *cache =
      detail::currentStructuredEvaluationCache();
  if (!cache)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_invalid: analytic priming requires an "
        "active invocation cache");
  if (structuredProgramReference.schemaIdentity !=
          frontend::structuredProgramArtifactSchema.identity ||
      structuredProgramReference.schemaVersion !=
          frontend::structuredProgramArtifactSchema.version ||
      structuredProgramReference.artifact != candidate.candidate.identity())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_invalid: cache subject mismatch");
  if (auto stored = artifactStore.get(structuredProgramReference); !stored)
    return stored.takeError();
  if (auto stored = artifactStore.get(fabricRoot.reference()); !stored)
    return stored.takeError();
  const sim::StructuredProgramSimulationWorkload *workload =
      invocation.simulationWorkload.structuredProgram();
  const sim::StructuredProgramSimulationRuntimeInput *runtimeInput =
      invocation.simulationRuntimeInput.structuredProgram();
  if (!workload || !runtimeInput ||
      invocation.simulationWorkload.identity() !=
          invocation.workload.artifact ||
      invocation.simulationRuntimeInput.identity() !=
          invocation.runtimeInput.artifact ||
      workload->entryRef.parent != invocation.sourceProgram.identity() ||
      runtimeInput->workloadIdentity !=
          invocation.simulationWorkload.identity())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_invalid: invocation input mismatch");
  if ((candidate.canonicalDataflow != nullptr) !=
      !candidate.spatialGraphs.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_invalid: candidate projection is partial");

  auto configView =
      ResolvedModelConfigView::project(kModelDescriptor.reference(), config);
  if (!configView)
    return configView.takeError();
  std::optional<dataflow::CanonicalDataflowProgramView> dataflowView;
  if (candidate.canonicalDataflow) {
    auto view = candidate.canonicalDataflow->view();
    if (!view)
      return view.takeError();
    dataflowView.emplace(std::move(*view));
  }
  auto sourceActivity = projectBlockActivity(invocation.sourceProgram,
                                             invocation.sourceObservations);
  if (!sourceActivity)
    return sourceActivity.takeError();
  std::optional<BlockActivityProjection> projectedActivity;
  const BlockActivityProjection *activity = nullptr;
  if (candidate.observations) {
    auto observed =
        projectBlockActivity(candidate.candidate, *candidate.observations);
    if (!observed)
      return observed.takeError();
    projectedActivity.emplace(std::move(*observed));
    activity = &*projectedActivity;
  } else if (candidate.candidate.identity() ==
             invocation.sourceProgram.identity()) {
    activity = &*sourceActivity;
  } else {
    if (candidate.blockActivityLineage.empty())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_fabric_model_invalid: candidate has no block activity "
          "lineage");
    auto projected = projectBlockActivity(candidate.candidate, *sourceActivity,
                                          candidate.blockActivityLineage);
    if (!projected)
      return projected.takeError();
    projectedActivity.emplace(std::move(*projected));
    activity = &*projectedActivity;
  }
  auto metrics = estimateMetrics(*activity, fabricRoot,
                                 dataflowView ? &*dataflowView : nullptr,
                                 candidate.spatialGraphs);
  if (!metrics)
    return metrics.takeError();
  const detail::StructuredAnalyticCacheKey key = metricCacheKey(
      structuredProgramReference, fabricRoot.reference(), invocation.workload,
      invocation.runtimeInput, configView->digest());
  auto &impl = detail::StructuredEvaluationCacheAccess::impl(*cache);
  auto cached = std::make_shared<const CachedMetrics>(*metrics);
  std::lock_guard<std::mutex> lock(impl.mutex);
  auto [found, inserted] =
      impl.analyticResults.try_emplace(key, std::move(cached));
  if (!inserted && *found->second != *metrics)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_fabric_model_invalid: nondeterministic cached result");
  if (inserted)
    impl.analyticPrimeCount.fetch_add(1, std::memory_order_relaxed);
  return llvm::Error::success();
}

llvm::Expected<std::optional<std::uint64_t>>
lookupStructuredFabricAnalyticRuntimeEstimate(
    const ArtifactRootReference &structuredProgram,
    const ArtifactRootReference &fabric, const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ResolvedConfig &config,
    StructuredEvaluationInvocationCache &cache) {
  auto configView =
      ResolvedModelConfigView::project(kModelDescriptor.reference(), config);
  if (!configView)
    return configView.takeError();
  const detail::StructuredAnalyticCacheKey key = metricCacheKey(
      structuredProgram, fabric, workload, runtimeInput, configView->digest());
  auto &impl = detail::StructuredEvaluationCacheAccess::impl(cache);
  std::lock_guard<std::mutex> lock(impl.mutex);
  auto found = impl.analyticResults.find(key);
  if (found == impl.analyticResults.end() || !*found->second)
    return std::optional<std::uint64_t>{};
  return std::optional<std::uint64_t>((**found->second).runtimePicoseconds);
}

llvm::Expected<bool> hasStructuredFabricAnalyticResult(
    const ArtifactRootReference &structuredProgram,
    const ArtifactRootReference &fabric, const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ResolvedConfig &config,
    StructuredEvaluationInvocationCache &cache) {
  auto configView =
      ResolvedModelConfigView::project(kModelDescriptor.reference(), config);
  if (!configView)
    return configView.takeError();
  const detail::StructuredAnalyticCacheKey key = metricCacheKey(
      structuredProgram, fabric, workload, runtimeInput, configView->digest());
  auto &impl = detail::StructuredEvaluationCacheAccess::impl(cache);
  std::lock_guard<std::mutex> lock(impl.mutex);
  return impl.analyticResults.find(key) != impl.analyticResults.end();
}

} // namespace loom::evaluation::models
