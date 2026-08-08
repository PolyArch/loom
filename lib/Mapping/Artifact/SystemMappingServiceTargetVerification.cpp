#include "SystemMappingServiceTargetVerification.h"

#include "Mapping/Artifact/SystemServiceBindingProjection.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace loom::mapping::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_mapping_service_target_invalid: " +
                                     message);
}

bool sameInterval(const SpatialMemoryIntervalView &left,
                  const SpatialMemoryIntervalView &right) {
  if (left.index() != right.index())
    return false;
  if (std::holds_alternative<SpatialMemoryWholeIntervalView>(left))
    return true;
  const auto &leftRange = std::get<SpatialMemoryByteRangeView>(left);
  const auto &rightRange = std::get<SpatialMemoryByteRangeView>(right);
  return leftRange.offsetBytes == rightRange.offsetBytes &&
         leftRange.sizeBytes == rightRange.sizeBytes;
}

struct BranchKey final {
  std::vector<std::uint8_t> region;
  std::vector<std::vector<std::uint8_t>> transforms;

  friend bool operator<(const BranchKey &left, const BranchKey &right) {
    return std::tie(left.region, left.transforms) <
           std::tie(right.region, right.transforms);
  }
  friend bool operator==(const BranchKey &left, const BranchKey &right) {
    return left.region == right.region && left.transforms == right.transforms;
  }
};

BranchKey branchKey(
    ::loom::fabric::FabricMemoryServiceRegionRef region,
    llvm::ArrayRef<::loom::fabric::SystemServiceTransformRef> transforms) {
  BranchKey result{::loom::fabric::canonicalFabricBytes(region), {}};
  result.transforms.reserve(transforms.size());
  for (const auto transform : transforms)
    result.transforms.push_back(
        ::loom::fabric::canonicalFabricBytes(transform));
  return result;
}

std::vector<BranchKey>
exactBranchKeys(const ::loom::fabric::FabricMemoryServiceTargetPlan &plan) {
  std::vector<BranchKey> result;
  result.reserve(plan.branches.size());
  for (const auto &branch : plan.branches)
    result.push_back(branchKey(branch.region, branch.transformPath));
  llvm::sort(result);
  return result;
}

std::vector<std::vector<std::uint8_t>>
regionKeys(const ::loom::fabric::FabricMemoryServiceTargetPlan &plan) {
  std::vector<std::vector<std::uint8_t>> result;
  result.reserve(plan.branches.size());
  for (const auto &branch : plan.branches)
    result.push_back(::loom::fabric::canonicalFabricBytes(branch.region));
  llvm::sort(result);
  return result;
}

std::vector<std::uint64_t>
selectedPlanOrdinals(const SystemServicePlanSelectionView &selection) {
  std::vector<std::uint64_t> result;
  for (const auto &clause : selection.clauses)
    result.push_back(clause.target);
  if (selection.defaultPlanOrdinal)
    result.push_back(*selection.defaultPlanOrdinal);
  llvm::sort(result);
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

const SystemServicePlanView *
findPlan(llvm::ArrayRef<SystemServicePlanView> plans, std::uint64_t ordinal) {
  auto found = llvm::find_if(
      plans, [&](const auto &plan) { return plan.ordinal == ordinal; });
  return found == plans.end() ? nullptr : &*found;
}

struct PlanMarks final {
  std::vector<bool> memoryTargets;
  std::vector<std::vector<bool>> exposures;
  std::vector<bool> consistencyTargets;
};

llvm::Expected<std::uint64_t>
moduleDependencyOrdinal(const ::loom::fabric::FabricSystemRootView &fabric,
                        ::loom::fabric::AccCoreOccurrenceRef core,
                        const SpatialMappingView &mapping) {
  auto target = fabric.spatialCoreTarget(core);
  if (!target ||
      target->dependencyOrdinal >= fabric.artifact().importedModules().size())
    return invalid("selected AccCore has no exact SpatialCore target");
  if (fabric.artifact()
          .importedModules()[target->dependencyOrdinal]
          .identity() != mapping.fabricIdentity())
    return invalid("selected SpatialMapping does not match the AccCore Module "
                   "target");
  return target->dependencyOrdinal;
}

llvm::Expected<
    std::pair<FinalizedSpatialMapping, SystemSpatialMemoryBindingProjection>>
resolveBinding(const ::loom::fabric::FabricSystemRootView &fabric,
               const ArtifactStore &store,
               const SpatialExecutionContextKey &context,
               const ServicePlanSelectionAnchor &anchor) {
  ArtifactRootReference reference{mappingArtifactSchema.identity.str(),
                                  mappingArtifactSchema.version,
                                  context.spatialMapping};
  auto mapping = importSpatialMapping(reference, store);
  if (!mapping)
    return mapping.takeError();
  auto dependency =
      moduleDependencyOrdinal(fabric, context.accCore, mapping->view());
  if (!dependency)
    return dependency.takeError();
  auto binding = projectSystemSpatialMemoryBinding(
      fabric, mapping->view(), *dependency, anchor, context.accCore);
  if (!binding)
    return binding.takeError();
  if (binding->endpointPairs.size() != 1)
    return invalid("selected execution does not resolve exactly one "
                   "attachment-bound memory endpoint pair");
  return std::make_pair(std::move(*mapping), std::move(*binding));
}

llvm::Expected<std::vector<::loom::fabric::FabricMemoryServiceTargetPlan>>
compatibleMemoryPlans(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                      const ::loom::fabric::FabricSystemRootView &fabric,
                      const SystemServiceObligationProjection &obligation,
                      const ServicePlanSelectionAnchor &anchor,
                      const SystemSpatialMemoryBindingProjection &binding) {
  const auto *operation =
      std::get_if<OperationServiceObligationFamilyKey>(&obligation.key);
  const auto *logicalMemory =
      operation ? std::get_if<::dataflow::LogicalMemoryRootOrViewRef>(operation)
                : nullptr;
  if (!logicalMemory || !binding.interval)
    return invalid("memory target anchor has no logical interval owner");
  const auto endpoint = binding.endpointPairs.front().systemEndpoint;
  auto plans = projectSystemMemoryTargetPlans(
      dataflow, fabric, endpoint, *logicalMemory, *binding.interval);
  if (!plans)
    return plans.takeError();

  std::optional<std::vector<::loom::fabric::FabricMemoryServiceRegionRef>>
      compatibleRegions;
  if (const auto *member =
          std::get_if<ServiceMemberPlanSelectionAnchor>(&anchor)) {
    auto regions = projectSystemOperationTargetRegions(
        dataflow, fabric, endpoint, member->member);
    if (!regions)
      return regions.takeError();
    compatibleRegions.emplace(std::move(*regions));
  }
  std::vector<::loom::fabric::FabricMemoryServiceTargetPlan> result;
  for (const auto &plan : *plans) {
    if (plan.branches.empty())
      return invalid("Fabric closure contains an empty memory target plan");
    if (!compatibleRegions ||
        llvm::all_of(plan.branches, [&](const auto &branch) {
          return llvm::is_contained(*compatibleRegions, branch.region);
        }))
      result.push_back(plan);
  }
  return result;
}

llvm::Error
verifyMemoryTarget(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                   const ::loom::fabric::FabricSystemRootView &fabric,
                   const SystemServiceObligationProjection &obligation,
                   const ServicePlanSelectionAnchor &anchor,
                   const SystemSpatialMemoryBindingProjection &binding,
                   const SystemServicePlanView &plan, PlanMarks &marks) {
  const auto *operation =
      std::get_if<OperationServiceObligationFamilyKey>(&obligation.key);
  const auto *logicalMemory =
      operation ? std::get_if<::dataflow::LogicalMemoryRootOrViewRef>(operation)
                : nullptr;
  if (!logicalMemory || !binding.interval)
    return invalid("memory target anchor has no logical interval owner");
  auto domain =
      compatibleMemoryPlans(dataflow, fabric, obligation, anchor, binding);
  if (!domain)
    return domain.takeError();

  std::vector<std::size_t> targetOrdinals;
  for (const auto &[ordinal, target] : llvm::enumerate(plan.memoryTargets))
    if (target.element.logicalMemory == *logicalMemory &&
        sameInterval(target.element.interval, *binding.interval))
      targetOrdinals.push_back(ordinal);
  if (targetOrdinals.empty())
    return invalid("selected memory plan omits its logical interval target");

  std::vector<std::vector<std::uint8_t>> selectedRegions;
  std::vector<BranchKey> selectedBranches;
  selectedRegions.reserve(targetOrdinals.size());
  selectedBranches.reserve(targetOrdinals.size());
  for (std::size_t ordinal : targetOrdinals) {
    const auto &target = plan.memoryTargets[ordinal];
    selectedRegions.push_back(
        ::loom::fabric::canonicalFabricBytes(target.element.serviceRegion));
    selectedBranches.push_back(
        branchKey(target.element.serviceRegion, target.element.transformPath));
  }
  llvm::sort(selectedRegions);
  llvm::sort(selectedBranches);

  std::vector<const ::loom::fabric::FabricMemoryServiceTargetPlan *>
      regionMatches;
  for (const auto &candidate : *domain)
    if (regionKeys(candidate) == selectedRegions)
      regionMatches.push_back(&candidate);
  if (regionMatches.empty())
    return invalid("selected service target is outside its attachment-bound "
                   "closure");
  if (regionMatches.size() == 1) {
    if (llvm::any_of(targetOrdinals, [&](std::size_t ordinal) {
          return !plan.memoryTargets[ordinal].element.transformPath.empty();
        }))
      return invalid("uniquely derived service transform path must be omitted");
  } else if (!llvm::any_of(regionMatches, [&](const auto *candidate) {
               return exactBranchKeys(*candidate) == selectedBranches;
             })) {
    return invalid("selected service target is outside its attachment-bound "
                   "closure");
  }

  const auto *exposure =
      std::get_if<MemoryExposurePlanSelectionAnchor>(&anchor);
  for (std::size_t ordinal : targetOrdinals) {
    marks.memoryTargets[ordinal] = true;
    if (!exposure)
      continue;
    if (!binding.exposureTerminal)
      return invalid("memory exposure has no Spatial provider terminal");
    std::optional<std::size_t> matched;
    for (const auto &[exposureOrdinal, child] :
         llvm::enumerate(plan.memoryTargets[ordinal].exposures)) {
      if (child.exposure != exposure->exposure)
        continue;
      if (matched || child.terminal != *binding.exposureTerminal)
        return invalid("memory exposure target has a duplicate or wrong "
                       "provider terminal");
      matched = exposureOrdinal;
    }
    if (!matched)
      return invalid("memory exposure target is incomplete");
    marks.exposures[ordinal][*matched] = true;
  }
  return llvm::Error::success();
}

llvm::Error verifyConsistencyTarget(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemServiceObligationProjection &obligation,
    const ServicePlanSelectionAnchor &anchor,
    const SystemSpatialMemoryBindingProjection &binding,
    const SystemServicePlanView &plan, PlanMarks &marks) {
  const auto *member = std::get_if<ServiceMemberPlanSelectionAnchor>(&anchor);
  if (!member ||
      !std::holds_alternative<::dataflow::FenceActorMemberRef>(member->member))
    return invalid("consistency target has a non-fence anchor");
  const auto endpoint = binding.endpointPairs.front().systemEndpoint;
  auto domains = projectSystemFenceTargetDomains(dataflow, fabric, endpoint,
                                                 member->member);
  if (!domains)
    return domains.takeError();
  const auto *operation =
      std::get_if<OperationServiceObligationFamilyKey>(&obligation.key);
  const auto *fence =
      operation ? std::get_if<::dataflow::FenceActorFamilyRef>(operation)
                : nullptr;
  if (!fence)
    return invalid("fence anchor belongs to a non-fence obligation");
  std::optional<std::size_t> matched;
  for (const auto &[ordinal, target] :
       llvm::enumerate(plan.consistencyTargets)) {
    if (target.fence != *fence)
      continue;
    if (matched || !llvm::is_contained(*domains, target.consistencyDomain))
      return invalid("selected consistency target is outside its "
                     "attachment-bound domain");
    matched = ordinal;
  }
  if (!matched)
    return invalid("selected fence plan has no compatible consistency target");
  marks.consistencyTargets[*matched] = true;
  return llvm::Error::success();
}

} // namespace

llvm::Error verifySystemServiceTargetClosure(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ArtifactStore &store,
    const SystemServiceObligationProjection &obligation,
    const SystemExecutionContextProjection &contexts,
    llvm::ArrayRef<SystemServicePlanView> plans,
    llvm::ArrayRef<SystemServicePlanSelectionView> selections) {
  if (std::holds_alternative<TransferObligationFamilyKey>(obligation.key))
    return llvm::Error::success();

  std::map<std::uint64_t, PlanMarks> marks;
  for (const auto &plan : plans) {
    PlanMarks planMarks;
    planMarks.memoryTargets.resize(plan.memoryTargets.size(), false);
    planMarks.exposures.reserve(plan.memoryTargets.size());
    for (const auto &target : plan.memoryTargets)
      planMarks.exposures.emplace_back(target.exposures.size(), false);
    planMarks.consistencyTargets.resize(plan.consistencyTargets.size(), false);
    marks.emplace(plan.ordinal, std::move(planMarks));
  }

  for (const auto &selection : selections) {
    const auto *context =
        std::get_if<SpatialExecutionContextKey>(&selection.key.context);
    if (!context)
      return invalid("operation service target has a non-Spatial execution "
                     "context");
    const bool reachable =
        llvm::any_of(contexts.spatialDomains, [&](const auto &domain) {
          return domain.context == *context;
        });
    if (!reachable)
      return invalid("operation service target has an unreachable execution "
                     "context");
    auto resolved =
        resolveBinding(fabric, store, *context, selection.key.anchor);
    if (!resolved)
      return resolved.takeError();
    const auto &binding = resolved->second;

    for (std::uint64_t ordinal : selectedPlanOrdinals(selection)) {
      const auto *plan = findPlan(plans, ordinal);
      auto mark = marks.find(ordinal);
      if (!plan || mark == marks.end())
        return invalid("service target selection names an absent plan");
      const auto *member =
          std::get_if<ServiceMemberPlanSelectionAnchor>(&selection.key.anchor);
      const bool fence =
          member && std::holds_alternative<::dataflow::FenceActorMemberRef>(
                        member->member);
      llvm::Error error =
          fence ? verifyConsistencyTarget(dataflow, fabric, obligation,
                                          selection.key.anchor, binding, *plan,
                                          mark->second)
                : verifyMemoryTarget(dataflow, fabric, obligation,
                                     selection.key.anchor, binding, *plan,
                                     mark->second);
      if (error)
        return error;
    }
  }

  for (const auto &plan : plans) {
    const auto &planMarks = marks.at(plan.ordinal);
    if (llvm::is_contained(planMarks.memoryTargets, false) ||
        llvm::is_contained(planMarks.consistencyTargets, false))
      return invalid("service plan contains a foreign target element");
    for (const auto &exposures : planMarks.exposures)
      if (llvm::is_contained(exposures, false))
        return invalid("service plan contains a foreign memory exposure");
  }
  return llvm::Error::success();
}

} // namespace loom::mapping::detail
