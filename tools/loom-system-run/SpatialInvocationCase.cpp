#include "SpatialInvocationCase.h"
#include "SystemRunError.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Deployment/DeploymentSpatialLaunchSelection.h"
#include "Evaluation/Models/CgraSimulation.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/SystemMappingIdentity.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SpatialInvocation.h"
#include "Simulator/SpatialObservationComparison.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <set>
#include <utility>
#include <variant>

namespace loom::system_run {
namespace {

bool sameValueSequence(const loom::sim::CanonicalValueSequence &lhs,
                       const loom::sim::CanonicalValueSequence &rhs) {
  return lhs.tokenCount == rhs.tokenCount && lhs.lanes == rhs.lanes;
}

bool sameStreamSequence(const loom::sim::CanonicalStreamSequence &lhs,
                        const loom::sim::CanonicalStreamSequence &rhs) {
  return lhs.termination == rhs.termination &&
         sameValueSequence(lhs.values, rhs.values);
}

bool sameInvocationPointerTarget(
    const std::optional<loom::runtime::SpatialInvocationPointerTarget> &lhs,
    const std::optional<loom::runtime::SpatialInvocationPointerTarget> &rhs) {
  if (lhs.has_value() != rhs.has_value())
    return false;
  return !lhs || (lhs->objectOrdinal == rhs->objectOrdinal &&
                  lhs->byteOffset == rhs->byteOffset);
}

bool sameInvocationValue(const loom::runtime::SpatialInvocationValue &lhs,
                         const loom::runtime::SpatialInvocationValue &rhs) {
  return lhs.ordinal == rhs.ordinal && lhs.bitCount == rhs.bitCount &&
         sameInvocationPointerTarget(lhs.pointerTarget, rhs.pointerTarget) &&
         lhs.littleEndianBits == rhs.littleEndianBits;
}

bool sameInvocationMemoryRootBinding(
    const loom::runtime::SpatialInvocationMemoryRootBinding &lhs,
    const loom::runtime::SpatialInvocationMemoryRootBinding &rhs) {
  return lhs.logicalMemoryRootEntity == rhs.logicalMemoryRootEntity &&
         lhs.objectOrdinal == rhs.objectOrdinal &&
         lhs.byteOffset == rhs.byteOffset;
}

bool sameInvocationResultDestination(
    const loom::runtime::SpatialInvocationResultDestination &lhs,
    const loom::runtime::SpatialInvocationResultDestination &rhs) {
  return lhs.ordinal == rhs.ordinal && lhs.bitCount == rhs.bitCount &&
         lhs.address == rhs.address;
}

bool sameRuntimeMemoryPointer(const loom::sim::RuntimeMemoryPointer &lhs,
                              const loom::sim::RuntimeMemoryPointer &rhs) {
  return lhs.storageByteOffset == rhs.storageByteOffset &&
         lhs.addressSpace == rhs.addressSpace && lhs.target == rhs.target;
}

bool sameRuntimeMemoryBinding(const loom::sim::MemoryRootBindingEntry &lhs,
                              const loom::sim::MemoryRootBindingEntry &rhs) {
  return lhs.root == rhs.root &&
         lhs.binding.objectOrdinal == rhs.binding.objectOrdinal &&
         lhs.binding.byteOffset == rhs.binding.byteOffset;
}

bool sameMemoryBytes(llvm::ArrayRef<loom::sim::SemanticMemoryByte> lhs,
                     llvm::ArrayRef<loom::sim::SemanticMemoryByte> rhs) {
  return lhs.size() == rhs.size() &&
         llvm::equal(lhs, rhs, [](const auto &left, const auto &right) {
           return left.state == right.state && left.value == right.value;
         });
}

llvm::Expected<std::set<std::uint64_t>>
readMemoryRoots(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                ::dataflow::RootedGraphLaunchRef launch) {
  std::set<std::uint64_t> roots;
  llvm::Error error = dataflow.forEachContextualServiceActor(
      launch.rootThreadLaunch,
      [&](::dataflow::ContextualActorRef actorRef) -> llvm::Error {
        if (actorRef.launch != launch)
          return llvm::Error::success();
        auto actor = dataflow.resolve(actorRef.actor);
        if (!actor)
          return actor.takeError();
        if (llvm::isa<::dataflow::FenceOp>(actor->op))
          return llvm::Error::success();
        auto access =
            ::dataflow::semantics::getCanonicalMemoryAccessView(actor->op);
        if (!access)
          return access.takeError();
        if (access->operation() ==
            ::dataflow::semantics::MemoryAccessOperation::Store)
          return llvm::Error::success();
        auto memory = dataflow.resolveAddressedMemory(actorRef);
        if (!memory)
          return memory.takeError();
        std::optional<dataflow::LogicalMemoryRootRef> root;
        if (const auto *rootReference =
                std::get_if<dataflow::LogicalMemoryRootRef>(&*memory))
          root = *rootReference;
        else
          root = std::get<dataflow::LogicalMemoryViewRef>(*memory).root;
        roots.insert(root->entity.value());
        return llvm::Error::success();
      });
  if (error)
    return std::move(error);
  return roots;
}

bool sameRuntimeInputSemantics(
    const loom::sim::SpatialSimulationRuntimeInput &lhs,
    const loom::sim::SpatialSimulationRuntimeInput &rhs,
    llvm::ArrayRef<std::uint8_t> compareMemoryBytes) {
  if (lhs.runtimeValues.size() != rhs.runtimeValues.size() ||
      lhs.runtimeStreams.size() != rhs.runtimeStreams.size() ||
      lhs.memoryObjects.size() != rhs.memoryObjects.size() ||
      lhs.workloadIdentity != rhs.workloadIdentity ||
      lhs.memoryRootBindings.size() != rhs.memoryRootBindings.size() ||
      !llvm::equal(lhs.memoryRootBindings, rhs.memoryRootBindings,
                   sameRuntimeMemoryBinding))
    return false;
  for (std::size_t ordinal = 0; ordinal != lhs.runtimeValues.size(); ++ordinal)
    if (lhs.runtimeValues[ordinal].valueInputOrdinal !=
            rhs.runtimeValues[ordinal].valueInputOrdinal ||
        !sameValueSequence(lhs.runtimeValues[ordinal].value,
                           rhs.runtimeValues[ordinal].value))
      return false;
  for (std::size_t ordinal = 0; ordinal != lhs.runtimeStreams.size(); ++ordinal)
    if (!sameStreamSequence(lhs.runtimeStreams[ordinal],
                            rhs.runtimeStreams[ordinal]))
      return false;
  for (std::size_t ordinal = 0; ordinal != lhs.memoryObjects.size();
       ++ordinal) {
    const auto &left = lhs.memoryObjects[ordinal];
    const auto &right = rhs.memoryObjects[ordinal];
    if (left.pointerValues.size() != right.pointerValues.size() ||
        !llvm::equal(left.pointerValues, right.pointerValues,
                     sameRuntimeMemoryPointer) ||
        left.initialBytes.size() != right.initialBytes.size() ||
        (compareMemoryBytes[ordinal] &&
         !sameMemoryBytes(left.initialBytes, right.initialBytes)))
      return false;
  }
  return true;
}

llvm::Expected<bool> sameInvocationSemantics(
    const loom::runtime::SpatialInvocationWire &lhs,
    const loom::runtime::SpatialInvocationWire &rhs,
    const loom::sim::SpatialSimulationRuntimeInput &lhsRuntime,
    const loom::sim::SpatialSimulationRuntimeInput &rhsRuntime,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    ::dataflow::RootedGraphLaunchRef launch) {
  if (lhs.canonicalDataflowIdentity != rhs.canonicalDataflowIdentity ||
      lhs.rootThreadLaunchEntity != rhs.rootThreadLaunchEntity ||
      lhs.graphLaunchEntity != rhs.graphLaunchEntity ||
      lhs.denseCoordinates != rhs.denseCoordinates ||
      lhs.values.size() != rhs.values.size() ||
      !llvm::equal(lhs.values, rhs.values, sameInvocationValue) ||
      lhs.memoryRootBindings.size() != rhs.memoryRootBindings.size() ||
      !llvm::equal(lhs.memoryRootBindings, rhs.memoryRootBindings,
                   sameInvocationMemoryRootBinding) ||
      lhs.results.size() != rhs.results.size() ||
      !llvm::equal(lhs.results, rhs.results, sameInvocationResultDestination) ||
      lhs.memoryObjects.size() != rhs.memoryObjects.size())
    return false;
  auto readRoots = readMemoryRoots(dataflow, launch);
  if (!readRoots)
    return readRoots.takeError();
  std::vector<std::uint8_t> compareMemoryBytes(lhs.memoryObjects.size(), 0);
  for (const auto &binding : lhs.memoryRootBindings)
    if (readRoots->find(binding.logicalMemoryRootEntity) != readRoots->end()) {
      if (binding.objectOrdinal >= compareMemoryBytes.size())
        return invalid("invocation memory binding exceeds its object table");
      compareMemoryBytes[binding.objectOrdinal] = 1;
    }
  for (std::size_t ordinal = 0; ordinal != lhs.memoryObjects.size();
       ++ordinal) {
    const auto &left = lhs.memoryObjects[ordinal];
    const auto &right = rhs.memoryObjects[ordinal];
    if (left.ordinal != right.ordinal || left.address != right.address ||
        left.initialBytes.size() != right.initialBytes.size() ||
        (compareMemoryBytes[ordinal] &&
         left.initialBytes != right.initialBytes))
      return false;
  }
  return sameRuntimeInputSemantics(lhsRuntime, rhsRuntime, compareMemoryBytes);
}

bool sameWrites(llvm::ArrayRef<loom::sim::SpatialInvocationMemoryWrite> lhs,
                llvm::ArrayRef<loom::sim::SpatialInvocationMemoryWrite> rhs) {
  return lhs.size() == rhs.size() &&
         llvm::equal(lhs, rhs, [](const auto &left, const auto &right) {
           return left.address == right.address && left.bytes == right.bytes;
         });
}

} // namespace

llvm::Expected<SpatialInvocationCase> materializeSpatialInvocationCase(
    std::size_t ordinal, const ObservedSpatialInvocation &dfg,
    const ObservedSpatialInvocation &cgra,
    const loom::deployment::FinalizedDeployment &deployment,
    const loom::ArtifactStore &artifacts, const loom::BlobStore &blobs) {
  if (dfg.dispatchTargetOrdinal != cgra.dispatchTargetOrdinal ||
      dfg.accCoreReference != cgra.accCoreReference ||
      dfg.executionContextKey != cgra.executionContextKey ||
      dfg.workload != cgra.workload)
    return invalid("System DFG and CGRA observed different effective "
                   "invocations");
  loom::runtime::SpatialInvocationWire dfgWire;
  std::string diagnostic;
  if (!loom::runtime::decodeSpatialInvocationWire(dfg.invocation, dfgWire,
                                                  diagnostic))
    return invalid("cannot decode System Spatial invocation: " +
                   llvm::Twine(diagnostic));
  loom::runtime::SpatialInvocationWire cgraWire;
  diagnostic.clear();
  if (!loom::runtime::decodeSpatialInvocationWire(cgra.invocation, cgraWire,
                                                  diagnostic))
    return invalid("cannot decode CGRA System Spatial invocation: " +
                   llvm::Twine(diagnostic));
  auto dataflowIdentity =
      loom::ArtifactIdentity::fromBytes(dfgWire.canonicalDataflowIdentity);
  if (!dataflowIdentity)
    return dataflowIdentity.takeError();
  loom::ArtifactRootReference dataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version, *dataflowIdentity};
  auto dfgWorkload =
      loom::sim::importSpatialSimulationWorkload(dfg.workload, artifacts);
  if (!dfgWorkload)
    return dfgWorkload.takeError();
  auto cgraWorkload =
      loom::sim::importSpatialSimulationWorkload(cgra.workload, artifacts);
  if (!cgraWorkload)
    return cgraWorkload.takeError();
  auto dfgDataflowView = dfgWorkload->dataflow.view();
  if (!dfgDataflowView)
    return dfgDataflowView.takeError();
  auto cgraDataflowView = cgraWorkload->dataflow.view();
  if (!cgraDataflowView)
    return cgraDataflowView.takeError();
  if (dfgWorkload->dataflow.identity() != dataflowReference.artifact ||
      cgraWorkload->dataflow.identity() != dataflowReference.artifact)
    return invalid("System invocation workload has a foreign Dataflow owner");
  const auto *dfgSpatialWorkload = dfgWorkload->workload.spatial();
  const auto *cgraSpatialWorkload = cgraWorkload->workload.spatial();
  if (!dfgSpatialWorkload || !cgraSpatialWorkload ||
      dfgSpatialWorkload->launchRef != cgraSpatialWorkload->launchRef)
    return invalid("System invocation workload is not Spatial");
  const dataflow::RootedGraphLaunchRef graph = dfgSpatialWorkload->launchRef;
  auto dfgRuntimeIdentity =
      loom::ArtifactIdentity::fromBytes(dfg.runtimeInput.identity);
  if (!dfgRuntimeIdentity)
    return dfgRuntimeIdentity.takeError();
  auto dfgRuntime = loom::sim::importSimulationRuntimeInput(
      dfg.runtimeInput.canonicalBytes, dfgWorkload->workload, *dfgDataflowView,
      *dfgRuntimeIdentity);
  if (!dfgRuntime)
    return dfgRuntime.takeError();
  auto cgraRuntimeIdentity =
      loom::ArtifactIdentity::fromBytes(cgra.runtimeInput.identity);
  if (!cgraRuntimeIdentity)
    return cgraRuntimeIdentity.takeError();
  auto cgraRuntime = loom::sim::importSimulationRuntimeInput(
      cgra.runtimeInput.canonicalBytes, cgraWorkload->workload,
      *cgraDataflowView, *cgraRuntimeIdentity);
  if (!cgraRuntime)
    return cgraRuntime.takeError();
  if (llvm::Error error =
          loom::sim::validateEffectiveSpatialInvocationRuntimeInput(
              *dfgWorkload, dfgWire, *dfgRuntime))
    return std::move(error);
  if (llvm::Error error =
          loom::sim::validateEffectiveSpatialInvocationRuntimeInput(
              *cgraWorkload, cgraWire, *cgraRuntime))
    return std::move(error);
  const auto *dfgSpatialRuntime = dfgRuntime->spatial();
  const auto *cgraSpatialRuntime = cgraRuntime->spatial();
  if (!dfgSpatialRuntime || !cgraSpatialRuntime)
    return invalid("System invocation runtime is not Spatial");
  auto invocationSemantics =
      sameInvocationSemantics(dfgWire, cgraWire, *dfgSpatialRuntime,
                              *cgraSpatialRuntime, *dfgDataflowView, graph);
  if (!invocationSemantics)
    return invocationSemantics.takeError();
  if (!*invocationSemantics)
    return invalid("System DFG and CGRA observed different effective "
                   "invocation semantics");
  loom::sim::ImportedSpatialSimulationInputs dfgInputs{
      std::move(dfgWorkload->dataflow), std::move(dfgWorkload->workload),
      std::move(*dfgRuntime)};
  loom::sim::ImportedSpatialSimulationInputs cgraInputs{
      std::move(cgraWorkload->dataflow), std::move(cgraWorkload->workload),
      std::move(*cgraRuntime)};
  auto runtimeReference = loom::sim::publishSimulationRuntimeInput(
      dfgInputs.runtimeInput, artifacts);
  if (!runtimeReference)
    return runtimeReference.takeError();
  auto selection = loom::deployment::resolveDeploymentSpatialLaunchSelection(
      deployment, graph, dfgWire.denseCoordinates, artifacts, blobs);
  if (!selection)
    return selection.takeError();
  const std::string selectedAccCoreReference =
      loom::formatArtifactLocalPayloadHex(
          loom::fabric::canonicalFabricBytes(selection->context.accCore));
  auto selectedContextBytes = loom::mapping::encodeExecutionContextKey(
      loom::mapping::ExecutionContextKey(selection->context));
  if (!selectedContextBytes)
    return selectedContextBytes.takeError();
  const std::string selectedContextKey =
      loom::formatArtifactLocalPayloadHex(*selectedContextBytes);
  if (dfg.accCoreReference != selectedAccCoreReference ||
      dfg.executionContextKey != selectedContextKey)
    return invalid("gem5 target differs from the Deployment execution context");
  auto cgraCase = loom::evaluation::models::resolveCgraSimulationCase(
      selection->spatialMapping, dfg.workload, *runtimeReference, artifacts);
  if (!cgraCase)
    return cgraCase.takeError();
  if (cgraCase->canonicalDataflow != dataflowReference)
    return invalid("Deployment selected a foreign Dataflow owner");

  auto dfgBoundary = loom::sim::decodeSpatialEngineBoundaryResult(
      dfg.boundaryResult, dfgInputs);
  if (!dfgBoundary)
    return dfgBoundary.takeError();
  auto cgraBoundary = loom::sim::decodeSpatialEngineBoundaryResult(
      cgra.boundaryResult, cgraInputs);
  if (!cgraBoundary)
    return cgraBoundary.takeError();
  if (!std::holds_alternative<loom::sim::RetiredExecution>(
          dfgBoundary->terminal) ||
      !std::holds_alternative<loom::sim::RetiredExecution>(
          cgraBoundary->terminal) ||
      !loom::sim::haveExactlyEqualSpatialFunctionalObservations(
          dfgBoundary->functionalObservations,
          cgraBoundary->functionalObservations))
    return invalid("System Spatial DFG and CGRA observations differ");
  auto dfgWrites = loom::sim::projectSpatialInvocationResultWrites(
      dfgWire, dfgInputs, dfgBoundary->functionalObservations);
  if (!dfgWrites)
    return dfgWrites.takeError();
  auto cgraWrites = loom::sim::projectSpatialInvocationResultWrites(
      cgraWire, cgraInputs, cgraBoundary->functionalObservations);
  if (!cgraWrites)
    return cgraWrites.takeError();
  if (!sameWrites(*dfgWrites, *cgraWrites))
    return invalid("System Spatial engines projected different guest writes");

  return SpatialInvocationCase{ordinal,
                               dfg.dispatchTargetOrdinal,
                               dfg.accCoreReference,
                               dfg.executionContextKey,
                               std::move(dfgWire.denseCoordinates),
                               std::move(dataflowReference),
                               dfg.workload,
                               std::move(*runtimeReference),
                               std::move(selection->hardwareImplementation),
                               std::move(cgraCase->fabric),
                               std::move(selection->spatialMapping),
                               std::move(*dfgBoundary),
                               std::move(*cgraBoundary)};
}

} // namespace loom::system_run
