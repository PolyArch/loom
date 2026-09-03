#include "PnR/System/SystemMappingMigration.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Mapping/Artifact/SystemMappingIdentity.h"

#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::pnr {
namespace {

template <typename Target>
bool sameClauses(
    llvm::ArrayRef<::loom::mapping::SystemPresburgerClauseView<Target>> lhs,
    llvm::ArrayRef<::loom::mapping::SystemPresburgerClauseView<Target>> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs, rhs))
    if (left.cells != right.cells || !(left.target == right.target))
      return false;
  return true;
}

template <typename Target>
bool sameStableEntries(
    llvm::ArrayRef<::loom::mapping::SystemStableKeyEntryView<Target>> lhs,
    llvm::ArrayRef<::loom::mapping::SystemStableKeyEntryView<Target>> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs, rhs))
    if (!(left.key == right.key) || !(left.target == right.target))
      return false;
  return true;
}

bool sameThreadBinding(
    const ::loom::mapping::SystemThreadExecutionBindingView &lhs,
    const ::loom::mapping::SystemThreadExecutionBindingView &rhs) {
  return lhs.key == rhs.key && lhs.relationKind == rhs.relationKind &&
         lhs.defaultTarget == rhs.defaultTarget &&
         sameClauses(llvm::ArrayRef(lhs.clauses),
                     llvm::ArrayRef(rhs.clauses)) &&
         sameStableEntries(llvm::ArrayRef(lhs.stableKeyEntries),
                           llvm::ArrayRef(rhs.stableKeyEntries));
}

bool sameGraphBinding(
    const ::loom::mapping::SystemGraphExecutionBindingView &lhs,
    const ::loom::mapping::SystemGraphExecutionBindingView &rhs) {
  return lhs.key == rhs.key && lhs.relationKind == rhs.relationKind &&
         lhs.defaultTarget == rhs.defaultTarget &&
         sameClauses(llvm::ArrayRef(lhs.clauses),
                     llvm::ArrayRef(rhs.clauses)) &&
         sameStableEntries(llvm::ArrayRef(lhs.stableKeyEntries),
                           llvm::ArrayRef(rhs.stableKeyEntries));
}

bool sameMemoryInterval(const ::loom::mapping::SpatialMemoryIntervalView &lhs,
                        const ::loom::mapping::SpatialMemoryIntervalView &rhs) {
  if (lhs.index() != rhs.index())
    return false;
  if (std::holds_alternative<::loom::mapping::SpatialMemoryWholeIntervalView>(
          lhs))
    return true;
  const auto &left = std::get<::loom::mapping::SpatialMemoryByteRangeView>(lhs);
  const auto &right =
      std::get<::loom::mapping::SpatialMemoryByteRangeView>(rhs);
  return left.offsetBytes == right.offsetBytes &&
         left.sizeBytes == right.sizeBytes;
}

bool sameServicePlanElement(
    const ::loom::mapping::SystemServicePlanElementView &lhs,
    const ::loom::mapping::SystemServicePlanElementView &rhs) {
  if (lhs.index() != rhs.index())
    return false;
  if (const auto *left =
          std::get_if<::loom::mapping::CanonicalServiceLegKey>(&lhs))
    return *left == std::get<::loom::mapping::CanonicalServiceLegKey>(rhs);
  if (const auto *left =
          std::get_if<::loom::mapping::SystemMemoryRegionElementView>(&lhs)) {
    const auto &right =
        std::get<::loom::mapping::SystemMemoryRegionElementView>(rhs);
    return left->logicalMemory == right.logicalMemory &&
           sameMemoryInterval(left->interval, right.interval) &&
           left->serviceRegion == right.serviceRegion &&
           left->transformPath == right.transformPath;
  }
  const auto &left =
      std::get<::loom::mapping::SystemConsistencyElementView>(lhs);
  const auto &right =
      std::get<::loom::mapping::SystemConsistencyElementView>(rhs);
  return left.fence == right.fence &&
         left.consistencyDomain == right.consistencyDomain;
}

bool sameTransferLeg(const ::loom::mapping::SystemTransferLegView &lhs,
                     const ::loom::mapping::SystemTransferLegView &rhs) {
  if (!(lhs.leg == rhs.leg) || !(lhs.rootEndpoint == rhs.rootEndpoint) ||
      lhs.nodes.size() != rhs.nodes.size() ||
      lhs.sinks.size() != rhs.sinks.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs.nodes, rhs.nodes))
    if (left.ordinal != right.ordinal ||
        left.parentOrdinal != right.parentOrdinal ||
        !(left.incomingTraversal == right.incomingTraversal))
      return false;
  for (auto [left, right] : llvm::zip(lhs.sinks, rhs.sinks))
    if (!(left.terminal == right.terminal) ||
        left.nodeOrdinal != right.nodeOrdinal)
      return false;
  return true;
}

bool sameMemoryTarget(
    const ::loom::mapping::SystemMemoryRegionTargetView &lhs,
    const ::loom::mapping::SystemMemoryRegionTargetView &rhs) {
  if (!sameServicePlanElement(lhs.element, rhs.element) ||
      lhs.exposures.size() != rhs.exposures.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs.exposures, rhs.exposures))
    if (!(left.exposure == right.exposure) ||
        !(left.terminal == right.terminal))
      return false;
  return true;
}

bool sameServicePlan(const ::loom::mapping::SystemServicePlanView &lhs,
                     const ::loom::mapping::SystemServicePlanView &rhs) {
  if (lhs.ordinal != rhs.ordinal ||
      lhs.transferLegs.size() != rhs.transferLegs.size() ||
      lhs.memoryTargets.size() != rhs.memoryTargets.size() ||
      lhs.consistencyTargets.size() != rhs.consistencyTargets.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs.transferLegs, rhs.transferLegs))
    if (!sameTransferLeg(left, right))
      return false;
  for (auto [left, right] : llvm::zip(lhs.memoryTargets, rhs.memoryTargets))
    if (!sameMemoryTarget(left, right))
      return false;
  for (auto [left, right] :
       llvm::zip(lhs.consistencyTargets, rhs.consistencyTargets))
    if (!(left.fence == right.fence) ||
        !(left.consistencyDomain == right.consistencyDomain))
      return false;
  return true;
}

bool sameServiceSelection(
    const ::loom::mapping::SystemServicePlanSelectionView &lhs,
    const ::loom::mapping::SystemServicePlanSelectionView &rhs) {
  return lhs.key == rhs.key && lhs.relationKind == rhs.relationKind &&
         lhs.defaultPlanOrdinal == rhs.defaultPlanOrdinal &&
         sameClauses(llvm::ArrayRef(lhs.clauses),
                     llvm::ArrayRef(rhs.clauses)) &&
         sameStableEntries(llvm::ArrayRef(lhs.stableKeyEntries),
                           llvm::ArrayRef(rhs.stableKeyEntries));
}

bool sameServiceRealization(
    const ::loom::mapping::SystemServiceRealizationView &lhs,
    const ::loom::mapping::SystemServiceRealizationView &rhs) {
  if (!(lhs.key == rhs.key) || lhs.plans.size() != rhs.plans.size() ||
      lhs.selections.size() != rhs.selections.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs.plans, rhs.plans))
    if (!sameServicePlan(left, right))
      return false;
  for (auto [left, right] : llvm::zip(lhs.selections, rhs.selections))
    if (!sameServiceSelection(left, right))
      return false;
  return true;
}

bool sameResourceOwner(const ::loom::mapping::SystemResourceOwnerView &lhs,
                       const ::loom::mapping::SystemResourceOwnerView &rhs) {
  if (lhs.index() != rhs.index())
    return false;
  if (const auto *left =
          std::get_if<::loom::mapping::SystemInstructionResourceOwnerView>(
              &lhs)) {
    const auto &right =
        std::get<::loom::mapping::SystemInstructionResourceOwnerView>(rhs);
    return left->root == right.root &&
           left->instructionContext == right.instructionContext;
  }
  const auto &left =
      std::get<::loom::mapping::SystemServicePlanResourceOwnerView>(lhs);
  const auto &right =
      std::get<::loom::mapping::SystemServicePlanResourceOwnerView>(rhs);
  return left.service == right.service &&
         left.planOrdinal == right.planOrdinal &&
         sameServicePlanElement(left.element, right.element);
}

bool sameEventPoint(const ::loom::mapping::SystemEventPointView &lhs,
                    const ::loom::mapping::SystemEventPointView &rhs) {
  return lhs.event == rhs.event && lhs.guaranteedOffset == rhs.guaranteedOffset;
}

bool sameResourceUse(const ::loom::mapping::SystemResourceUseView &lhs,
                     const ::loom::mapping::SystemResourceUseView &rhs) {
  if (!sameResourceOwner(lhs.owner, rhs.owner) ||
      !(lhs.useSite == rhs.useSite) ||
      !sameEventPoint(lhs.activation.trigger, rhs.activation.trigger) ||
      lhs.activation.release.size() != rhs.activation.release.size() ||
      !(lhs.parameters == rhs.parameters) ||
      !(lhs.sharingAssignments == rhs.sharingAssignments))
    return false;
  for (auto [left, right] :
       llvm::zip(lhs.activation.release, rhs.activation.release))
    if (!sameEventPoint(left, right))
      return false;
  return true;
}

template <typename Value, typename Predicate, typename Equal>
bool sameFiltered(llvm::ArrayRef<Value> lhs, llvm::ArrayRef<Value> rhs,
                  Predicate &&keep, Equal &&equal) {
  std::vector<const Value *> left;
  std::vector<const Value *> right;
  for (const Value &value : lhs)
    if (keep(value))
      left.push_back(&value);
  for (const Value &value : rhs)
    if (keep(value))
      right.push_back(&value);
  if (left.size() != right.size())
    return false;
  for (auto [leftValue, rightValue] : llvm::zip(left, right))
    if (!equal(*leftValue, *rightValue))
      return false;
  return true;
}

bool containsObligation(
    llvm::ArrayRef<::loom::mapping::SystemServiceObligationProjection>
        obligations,
    const ::loom::mapping::SystemServiceObligationKey &key) {
  return llvm::any_of(obligations,
                      [&](const auto &value) { return value.key == key; });
}

::dataflow::RootThreadLaunchRef
rootOf(const ::dataflow::RootThreadBoundaryTransferRef &transfer) {
  return std::visit([](const auto &value) { return value.launch; }, transfer);
}

::dataflow::RootThreadLaunchRef
rootOf(const ::dataflow::GraphLaunchBoundaryTransferRef &transfer) {
  return std::visit(
      [](const auto &value) { return value.launch.rootThreadLaunch; },
      transfer);
}

::dataflow::RootThreadLaunchRef
rootOf(const ::dataflow::CanonicalProducerTerminalRef &terminal) {
  return std::visit(
      [](const auto &value) -> ::dataflow::RootThreadLaunchRef {
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value,
                                     ::dataflow::RootThreadBoundarySourceRef>)
          return rootOf(value.transfer);
        else if constexpr (std::is_same_v<
                               Value, ::dataflow::GraphLaunchBoundarySourceRef>)
          return rootOf(value.transfer);
        else
          return std::visit(
              [](const auto &producer) -> ::dataflow::RootThreadLaunchRef {
                using Producer = std::decay_t<decltype(producer)>;
                if constexpr (std::is_same_v<
                                  Producer,
                                  ::dataflow::GraphStreamOutputProducerRef>)
                  return producer.launch.rootThreadLaunch;
                else
                  return producer.launch;
              },
              value.producer);
      },
      terminal);
}

::dataflow::RootThreadLaunchRef
rootOf(const ::dataflow::CanonicalSinkTerminalRef &terminal) {
  return std::visit(
      [](const auto &value) -> ::dataflow::RootThreadLaunchRef {
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value,
                                     ::dataflow::RootThreadBoundarySinkRef>)
          return rootOf(value.transfer);
        else if constexpr (std::is_same_v<
                               Value, ::dataflow::GraphLaunchBoundarySinkRef>)
          return rootOf(value.transfer);
        else
          return std::visit(
              [](const auto &consumer) -> ::dataflow::RootThreadLaunchRef {
                using Consumer = std::decay_t<decltype(consumer)>;
                if constexpr (std::is_same_v<
                                  Consumer,
                                  ::dataflow::GraphStreamInputConsumerRef>)
                  return consumer.launch.rootThreadLaunch;
                else
                  return consumer.launch;
              },
              value.consumer);
      },
      terminal);
}

std::optional<::dataflow::RootThreadLaunchRef>
rootOf(const ::dataflow::ServiceMemberRef &member) {
  return std::visit(
      [](const auto &value) -> std::optional<::dataflow::RootThreadLaunchRef> {
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value,
                                     ::dataflow::MessageTransferMemberRef>)
          return std::nullopt;
        else
          return value.actor.launch.rootThreadLaunch;
      },
      member);
}

bool obligationTouchesPreservedRoot(
    const ::loom::mapping::SystemServiceObligationProjection &obligation,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> preservedRoots) {
  const auto preserved = [&](::dataflow::RootThreadLaunchRef root) {
    return llvm::is_contained(preservedRoots, root);
  };
  if (const auto *transfer =
          std::get_if<::loom::mapping::TransferObligationFamilyKey>(
              &obligation.key))
    if (preserved(rootOf(*transfer)))
      return true;
  if (llvm::any_of(obligation.members,
                   [&](const auto &member) {
                     const auto root = rootOf(member);
                     return root && preserved(*root);
                   }) ||
      llvm::any_of(obligation.sinks,
                   [&](const auto &sink) { return preserved(rootOf(sink)); }) ||
      llvm::any_of(obligation.exposures, [&](const auto &exposure) {
        return preserved(exposure.launch.rootThreadLaunch);
      }))
    return true;
  return false;
}

} // namespace

llvm::Expected<bool> preservesSystemMappingMigrationCone(
    const ::loom::mapping::SystemMappingView &parent,
    const ::loom::mapping::SystemMappingView &child,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> reopenedRoots,
    const ArtifactStore &store) {
  if (parent.dataflowIdentity() != child.dataflowIdentity() ||
      parent.fabricIdentity() != child.fabricIdentity())
    return false;
  auto parentCone =
      projectSystemMappingMigrationConePartition(parent, reopenedRoots, store);
  if (!parentCone)
    return parentCone.takeError();
  auto childCone =
      projectSystemMappingMigrationConePartition(child, reopenedRoots, store);
  if (!childCone)
    return childCone.takeError();
  if (parentCone->preservedGraphs != childCone->preservedGraphs ||
      parentCone->preservedTechMappings != childCone->preservedTechMappings ||
      parentCone->preservedSpatialMappings !=
          childCone->preservedSpatialMappings)
    return false;

  const auto &parentExecution = parent.executionBindings();
  const auto &childExecution = child.executionBindings();
  if (parentExecution.rootThreadLaunches() !=
      childExecution.rootThreadLaunches())
    return false;
  std::vector<::dataflow::RootThreadLaunchRef> preservedRoots;
  for (const auto root : parentExecution.rootThreadLaunches())
    if (!llvm::is_contained(reopenedRoots, root))
      preservedRoots.push_back(root);
  const auto preservesRoot = [&](::dataflow::RootThreadLaunchRef root) {
    return llvm::is_contained(preservedRoots, root);
  };
  if (!sameFiltered(
          parentExecution.threadBindings(), childExecution.threadBindings(),
          [&](const auto &binding) { return preservesRoot(binding.key); },
          sameThreadBinding) ||
      !sameFiltered(
          parentExecution.graphBindings(), childExecution.graphBindings(),
          [&](const auto &binding) {
            return preservesRoot(binding.key.rootThreadLaunch);
          },
          sameGraphBinding))
    return false;
  if (preservedRoots.empty())
    return true;

  const ArtifactRootReference dataflowReference{
      ::dataflow::canonicalDataflowSchema.identity.str(),
      ::dataflow::canonicalDataflowSchema.version, parent.dataflowIdentity()};
  auto dataflowArtifact =
      ::dataflow::importCanonicalDataflow(dataflowReference, store);
  if (!dataflowArtifact)
    return dataflowArtifact.takeError();
  auto dataflow = dataflowArtifact->view();
  if (!dataflow)
    return dataflow.takeError();
  auto allObligations = ::loom::mapping::projectSystemServiceObligations(
      *dataflow, parentExecution.rootThreadLaunches());
  if (!allObligations)
    return allObligations.takeError();
  std::vector<::loom::mapping::SystemServiceObligationProjection>
      preservedObligations;
  for (const auto &obligation : *allObligations)
    if (obligationTouchesPreservedRoot(obligation, preservedRoots))
      preservedObligations.push_back(obligation);
  const auto preservesObligation = [&](const auto &key) {
    return containsObligation(preservedObligations, key);
  };
  if (!sameFiltered(
          parent.serviceRealizations(), child.serviceRealizations(),
          [&](const auto &service) { return preservesObligation(service.key); },
          sameServiceRealization))
    return false;
  const auto preservesUse = [&](const ::loom::mapping::SystemResourceUseView
                                    &use) {
    if (const auto *instruction =
            std::get_if<::loom::mapping::SystemInstructionResourceOwnerView>(
                &use.owner))
      return preservesRoot(instruction->root);
    return preservesObligation(
        std::get<::loom::mapping::SystemServicePlanResourceOwnerView>(use.owner)
            .service);
  };
  return sameFiltered(parent.resourceUses(), child.resourceUses(), preservesUse,
                      sameResourceUse);
}

} // namespace loom::pnr
