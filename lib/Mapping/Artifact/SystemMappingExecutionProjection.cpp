#include "Mapping/Artifact/SystemMappingExecutionProjection.h"

#include "Common/ArtifactLocalReference.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <map>
#include <string>
#include <utility>

namespace loom::mapping {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_mapping_execution_invalid: " +
                                     message);
}

std::string byteKey(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

template <typename Ref>
llvm::Expected<std::string> dataflowKey(const ArtifactIdentity &owner,
                                        const Ref &reference) {
  auto bytes = ::dataflow::encodeDataflowReference(owner, reference);
  if (!bytes)
    return bytes.takeError();
  return byteKey(*bytes);
}

llvm::Expected<SystemPresburgerCell>
legalDomain(const ::dataflow::CanonicalDataflowProgramView &dataflow,
            ::dataflow::RootThreadLaunchRef root) {
  auto logical = dataflow.projectRootThreadLogicalDomain(root);
  if (!logical)
    return logical.takeError();
  if (logical->kind == ::dataflow::ThreadDomainKind::DynamicWork) {
    auto dynamic = dataflow.projectDynamicWork(root);
    if (!dynamic)
      return dynamic.takeError();
    if (dynamic->stableItemKeys.size() != 1)
      return invalid("DynamicWork stable-key domain is not singleton");
    return canonicalizeSystemPresburgerCell(SystemPresburgerCell{});
  }
  SystemPresburgerCell cell;
  cell.dimensionCount = logical->coordinateRank;
  cell.symbolCount = logical->launchParameters.size();
  const std::size_t width =
      static_cast<std::size_t>(cell.dimensionCount) + cell.symbolCount + 1;
  for (std::uint32_t coordinate = 0; coordinate < cell.dimensionCount;
       ++coordinate) {
    std::vector<std::int64_t> lower(width, 0);
    lower[coordinate] = 1;
    cell.inequalities.push_back(std::move(lower));
    std::vector<std::int64_t> upper(width, 0);
    upper[coordinate] = -1;
    upper[cell.dimensionCount + coordinate] = 1;
    upper.back() = -1;
    cell.inequalities.push_back(std::move(upper));
  }
  return canonicalizeSystemPresburgerCell(cell);
}

template <typename Target> struct ConcreteRelationPart final {
  Target target;
  std::vector<SystemPresburgerCell> cells;
  std::vector<::dataflow::DynamicWorkStableItemKey> stableItemKeys;
};

template <typename Target, typename Key>
llvm::Expected<std::vector<ConcreteRelationPart<Target>>> concretizeRelation(
    const std::vector<SystemPresburgerClauseView<Target>> &clauses,
    const std::optional<Target> &defaultTarget,
    const SystemPresburgerCell &domain, Key &&targetKey) {
  std::map<std::string, ConcreteRelationPart<Target>> grouped;
  std::vector<SystemPresburgerCell> explicitCells;
  for (const auto &clause : clauses) {
    explicitCells.insert(explicitCells.end(), clause.cells.begin(),
                         clause.cells.end());
    auto key = targetKey(clause.target);
    if (!key)
      return key.takeError();
    auto [entry, inserted] = grouped.try_emplace(
        *key, ConcreteRelationPart<Target>{clause.target, {}, {}});
    entry->second.cells.insert(entry->second.cells.end(), clause.cells.begin(),
                               clause.cells.end());
  }
  if (defaultTarget) {
    auto complement = splitSystemPresburgerSet({domain}, explicitCells);
    if (!complement)
      return complement.takeError();
    if (complement->outside.empty())
      return invalid("binding default has an empty complement");
    auto key = targetKey(*defaultTarget);
    if (!key)
      return key.takeError();
    auto [entry, inserted] = grouped.try_emplace(
        *key, ConcreteRelationPart<Target>{*defaultTarget, {}, {}});
    entry->second.cells.insert(
        entry->second.cells.end(),
        std::make_move_iterator(complement->outside.begin()),
        std::make_move_iterator(complement->outside.end()));
  }
  std::vector<ConcreteRelationPart<Target>> result;
  result.reserve(grouped.size());
  for (auto &[key, part] : grouped) {
    (void)key;
    result.push_back(std::move(part));
  }
  return result;
}

template <typename Target, typename Key>
llvm::Expected<std::vector<ConcreteRelationPart<Target>>>
concretizeStableRelation(
    const std::vector<SystemStableKeyEntryView<Target>> &entries,
    Key &&targetKey) {
  std::map<std::string, ConcreteRelationPart<Target>> grouped;
  for (const auto &entry : entries) {
    auto key = targetKey(entry.target);
    if (!key)
      return key.takeError();
    auto [part, inserted] = grouped.try_emplace(
        *key, ConcreteRelationPart<Target>{entry.target, {}, {}});
    part->second.stableItemKeys.push_back(entry.key);
  }
  auto surrogate = canonicalizeSystemPresburgerCell(SystemPresburgerCell{});
  if (!surrogate)
    return surrogate.takeError();
  std::vector<ConcreteRelationPart<Target>> result;
  result.reserve(grouped.size());
  for (auto &[key, part] : grouped) {
    (void)key;
    part.cells.push_back(*surrogate);
    result.push_back(std::move(part));
  }
  return result;
}

template <typename Target> std::string fabricKey(const Target &target) {
  const auto bytes = ::loom::fabric::canonicalFabricBytes(target);
  return byteKey(bytes);
}

llvm::Expected<std::string> mappingKey(const ArtifactRootReference &reference) {
  return byteKey(encodeArtifactRootReference(reference));
}

std::vector<::dataflow::DynamicWorkStableItemKey> intersectStableItemKeys(
    llvm::ArrayRef<::dataflow::DynamicWorkStableItemKey> lhs,
    llvm::ArrayRef<::dataflow::DynamicWorkStableItemKey> rhs) {
  std::vector<::dataflow::DynamicWorkStableItemKey> intersection;
  for (const auto key : lhs)
    if (llvm::is_contained(rhs, key) && !llvm::is_contained(intersection, key))
      intersection.push_back(key);
  return intersection;
}

struct ThreadProjection final {
  ::dataflow::RootThreadLaunchRef root;
  ::mapping::SystemBindingRelationKind relationKind =
      ::mapping::SystemBindingRelationKind::PresburgerPartition;
  std::vector<ConcreteRelationPart<::loom::fabric::AccCoreOccurrenceRef>> parts;
};

} // namespace

llvm::Expected<SystemExecutionContextProjection> projectSystemExecutionContexts(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const SystemExecutionBindingView &execution) {
  SystemExecutionContextProjection result;
  std::map<std::string, ThreadProjection> threads;
  for (const auto &binding : execution.threadBindings()) {
    auto rootKey = dataflowKey(dataflow.identity(), binding.key);
    auto domain = legalDomain(dataflow, binding.key);
    if (!rootKey)
      return rootKey.takeError();
    if (!domain)
      return domain.takeError();
    auto parts =
        binding.relationKind ==
                ::mapping::SystemBindingRelationKind::StableKeyLookup
            ? concretizeStableRelation(
                  binding.stableKeyEntries,
                  [](const auto &target) -> llvm::Expected<std::string> {
                    return fabricKey(target);
                  })
            : concretizeRelation(
                  binding.clauses, binding.defaultTarget, *domain,
                  [](const auto &target) -> llvm::Expected<std::string> {
                    return fabricKey(target);
                  });
    if (!parts)
      return parts.takeError();
    for (const auto &part : *parts)
      result.instructionDomains.push_back(SystemInstructionContextDomain{
          binding.key, InstructionExecutionContextKey{part.target}, part.cells,
          binding.relationKind, part.stableItemKeys});
    if (!threads
             .emplace(*rootKey,
                      ThreadProjection{binding.key, binding.relationKind,
                                       std::move(*parts)})
             .second)
      return invalid("duplicate thread binding in execution projection");
  }

  for (const auto &binding : execution.graphBindings()) {
    auto rootKey =
        dataflowKey(dataflow.identity(), binding.key.rootThreadLaunch);
    auto domain = legalDomain(dataflow, binding.key.rootThreadLaunch);
    if (!rootKey)
      return rootKey.takeError();
    if (!domain)
      return domain.takeError();
    auto thread = threads.find(*rootKey);
    if (thread == threads.end())
      return invalid("graph binding has no parent thread binding");
    if (binding.relationKind != thread->second.relationKind)
      return invalid("graph and thread relation kinds differ");
    auto graphParts =
        binding.relationKind ==
                ::mapping::SystemBindingRelationKind::StableKeyLookup
            ? concretizeStableRelation(
                  binding.stableKeyEntries,
                  [](const auto &target) { return mappingKey(target); })
            : concretizeRelation(
                  binding.clauses, binding.defaultTarget, *domain,
                  [](const auto &target) { return mappingKey(target); });
    if (!graphParts)
      return graphParts.takeError();

    std::map<std::string, SystemSpatialContextDomain> grouped;
    for (const auto &graphPart : *graphParts)
      for (const auto &threadPart : thread->second.parts) {
        SpatialExecutionContextKey context{threadPart.target,
                                           graphPart.target.artifact};
        auto encoded = encodeExecutionContextKey(context);
        if (!encoded)
          return encoded.takeError();
        if (binding.relationKind ==
            ::mapping::SystemBindingRelationKind::StableKeyLookup) {
          auto commonKeys = intersectStableItemKeys(graphPart.stableItemKeys,
                                                    threadPart.stableItemKeys);
          if (commonKeys.empty())
            continue;
          auto surrogate =
              canonicalizeSystemPresburgerCell(SystemPresburgerCell{});
          if (!surrogate)
            return surrogate.takeError();
          grouped.try_emplace(byteKey(*encoded), SystemSpatialContextDomain{
                                                     binding.key,
                                                     graphPart.target,
                                                     context,
                                                     {*surrogate},
                                                     binding.relationKind,
                                                     std::move(commonKeys)});
          continue;
        }
        for (const auto &graphCell : graphPart.cells)
          for (const auto &threadCell : threadPart.cells) {
            auto overlap =
                intersectSystemPresburgerCells(graphCell, threadCell);
            if (!overlap)
              return overlap.takeError();
            if (!*overlap)
              continue;
            auto [entry, inserted] = grouped.try_emplace(
                byteKey(*encoded),
                SystemSpatialContextDomain{binding.key,
                                           graphPart.target,
                                           context,
                                           {},
                                           binding.relationKind,
                                           graphPart.stableItemKeys});
            entry->second.cells.push_back(std::move(**overlap));
          }
      }
    if (grouped.empty())
      return invalid("graph binding has no reachable execution context");
    for (auto &[key, projection] : grouped) {
      (void)key;
      result.spatialDomains.push_back(std::move(projection));
    }
  }
  return result;
}

llvm::Expected<InstructionExecutionContextKey>
selectSystemDynamicWorkInstructionExecutionContext(
    const SystemExecutionContextProjection &projection,
    ::dataflow::RootThreadLaunchRef root,
    ::dataflow::DynamicWorkStableItemKey stableItem) {
  std::map<std::string, InstructionExecutionContextKey> selected;
  bool foundRoot = false;
  for (const SystemInstructionContextDomain &domain :
       projection.instructionDomains) {
    if (domain.root != root)
      continue;
    foundRoot = true;
    if (domain.relationKind !=
        ::mapping::SystemBindingRelationKind::StableKeyLookup)
      return invalid("DynamicWork root has a non-stable execution binding");
    if (!llvm::is_contained(domain.stableItemKeys, stableItem))
      continue;
    auto key = encodeExecutionContextKey(domain.context);
    if (!key)
      return key.takeError();
    selected.try_emplace(byteKey(*key), domain.context);
  }
  if (!foundRoot)
    return invalid("DynamicWork root has no System execution binding");
  if (selected.empty())
    return invalid("DynamicWork stable item selects no Instruction context");
  if (selected.size() != 1)
    return invalid("DynamicWork stable item selects multiple Instruction "
                   "contexts");
  return selected.begin()->second;
}

llvm::Expected<SelectedSystemSpatialContext>
selectSystemDynamicWorkSpatialExecutionContext(
    const SystemExecutionContextProjection &projection,
    ::dataflow::RootedGraphLaunchRef graph,
    ::dataflow::DynamicWorkStableItemKey stableItem) {
  std::map<std::string, SelectedSystemSpatialContext> selected;
  bool foundGraph = false;
  for (const SystemSpatialContextDomain &domain : projection.spatialDomains) {
    if (domain.graph != graph)
      continue;
    foundGraph = true;
    if (domain.relationKind !=
        ::mapping::SystemBindingRelationKind::StableKeyLookup)
      return invalid("DynamicWork graph has a non-stable execution binding");
    if (!llvm::is_contained(domain.stableItemKeys, stableItem))
      continue;
    auto key = encodeExecutionContextKey(domain.context);
    if (!key)
      return key.takeError();
    selected.try_emplace(
        byteKey(*key),
        SelectedSystemSpatialContext{domain.spatialMapping, domain.context});
  }
  if (!foundGraph)
    return invalid("DynamicWork graph has no System execution binding");
  if (selected.empty())
    return invalid("DynamicWork stable item selects no Spatial context");
  if (selected.size() != 1)
    return invalid("DynamicWork stable item selects multiple Spatial contexts");
  return selected.begin()->second;
}

llvm::Expected<std::vector<fabric::SpatialCoreOccurrenceRef>>
projectSystemExecutionSpatialCoreSubjects(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const SystemExecutionBindingView &execution) {
  auto projection = projectSystemExecutionContexts(dataflow, execution);
  if (!projection)
    return projection.takeError();

  std::vector<fabric::SpatialCoreOccurrenceRef> subjects;
  subjects.reserve(projection->instructionDomains.size() +
                   projection->spatialDomains.size());
  for (const SystemInstructionContextDomain &domain :
       projection->instructionDomains)
    subjects.push_back(
        fabric::SpatialCoreOccurrenceRef{domain.context.accCore});
  for (const SystemSpatialContextDomain &domain : projection->spatialDomains)
    subjects.push_back(
        fabric::SpatialCoreOccurrenceRef{domain.context.accCore});
  llvm::sort(subjects, [](fabric::SpatialCoreOccurrenceRef lhs,
                          fabric::SpatialCoreOccurrenceRef rhs) {
    return fabric::canonicalFabricBytes(lhs) <
           fabric::canonicalFabricBytes(rhs);
  });
  subjects.erase(std::unique(subjects.begin(), subjects.end()), subjects.end());
  return subjects;
}

llvm::Expected<SelectedSystemSpatialContext>
selectSystemSpatialExecutionContext(
    const SystemExecutionContextProjection &projection,
    ::dataflow::RootedGraphLaunchRef graph,
    llvm::ArrayRef<std::uint64_t> denseCoordinates) {
  std::map<std::string, SelectedSystemSpatialContext> selected;
  bool foundGraph = false;
  for (const SystemSpatialContextDomain &domain : projection.spatialDomains) {
    if (domain.graph != graph)
      continue;
    foundGraph = true;
    if (domain.relationKind ==
        ::mapping::SystemBindingRelationKind::StableKeyLookup)
      return invalid("DynamicWork graph selection requires a stable item key");
    for (const SystemPresburgerCell &cell : domain.cells) {
      if (cell.dimensionCount != denseCoordinates.size())
        return invalid("launch coordinate rank does not match its domain");
      SystemPresburgerCell point;
      point.dimensionCount = cell.dimensionCount;
      point.symbolCount = cell.symbolCount;
      const std::size_t width = static_cast<std::size_t>(point.dimensionCount) +
                                point.symbolCount + 1;
      point.equalities.reserve(denseCoordinates.size());
      for (std::size_t dimension = 0; dimension < denseCoordinates.size();
           ++dimension) {
        if (denseCoordinates[dimension] >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max()))
          return invalid("launch coordinate exceeds the signed Presburger "
                         "domain");
        std::vector<std::int64_t> equality(width, 0);
        equality[dimension] = 1;
        equality.back() =
            -static_cast<std::int64_t>(denseCoordinates[dimension]);
        point.equalities.push_back(std::move(equality));
      }
      auto overlap = intersectSystemPresburgerCells(cell, point);
      if (!overlap)
        return overlap.takeError();
      if (!*overlap)
        continue;
      auto key = encodeExecutionContextKey(domain.context);
      if (!key)
        return key.takeError();
      selected.try_emplace(
          byteKey(*key),
          SelectedSystemSpatialContext{domain.spatialMapping, domain.context});
    }
  }
  if (!foundGraph)
    return invalid("graph launch has no System execution binding");
  if (selected.empty())
    return invalid("launch coordinate selects no Spatial execution context");
  if (selected.size() != 1)
    return invalid("launch coordinate has an ambiguous Spatial execution "
                   "context across legal symbol valuations");
  return selected.begin()->second;
}

llvm::Expected<std::uint64_t> selectSystemServicePlanOrdinal(
    const SystemServiceRealizationView &realization,
    const ServicePlanSelectionAnchor &anchor,
    const ExecutionContextKey &context,
    llvm::ArrayRef<SystemPresburgerCell> contextDomain,
    llvm::ArrayRef<std::uint64_t> denseCoordinates) {
  const SystemServicePlanSelectionView *selection = nullptr;
  for (const SystemServicePlanSelectionView &candidate :
       realization.selections) {
    if (!(candidate.key.anchor == anchor) ||
        !(candidate.key.context == context))
      continue;
    if (selection)
      return invalid("service plan selection key is duplicated");
    selection = &candidate;
  }
  if (!selection)
    return invalid("service plan selection key is absent");
  if (selection->relationKind ==
      ::mapping::SystemBindingRelationKind::StableKeyLookup)
    return invalid("DynamicWork service selection requires a stable item key");
  if (contextDomain.empty())
    return invalid("service plan selection context domain is empty");

  std::vector<SystemPresburgerCell> explicitCells;
  for (const auto &clause : selection->clauses)
    explicitCells.insert(explicitCells.end(), clause.cells.begin(),
                         clause.cells.end());
  std::vector<SystemPresburgerCell> defaultCells;
  if (selection->defaultPlanOrdinal) {
    auto complement = splitSystemPresburgerSet(contextDomain, explicitCells);
    if (!complement)
      return complement.takeError();
    defaultCells = std::move(complement->outside);
  }

  std::set<std::uint64_t> selected;
  const auto intersectsPoint =
      [&](const SystemPresburgerCell &cell) -> llvm::Expected<bool> {
    if (cell.dimensionCount != denseCoordinates.size())
      return invalid("service plan coordinate rank does not match its domain");
    SystemPresburgerCell point;
    point.dimensionCount = cell.dimensionCount;
    point.symbolCount = cell.symbolCount;
    const std::size_t width =
        static_cast<std::size_t>(point.dimensionCount) + point.symbolCount + 1;
    for (std::size_t dimension = 0; dimension < denseCoordinates.size();
         ++dimension) {
      if (denseCoordinates[dimension] >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
        return invalid("service plan coordinate exceeds the signed "
                       "Presburger domain");
      std::vector<std::int64_t> equality(width, 0);
      equality[dimension] = 1;
      equality.back() = -static_cast<std::int64_t>(denseCoordinates[dimension]);
      point.equalities.push_back(std::move(equality));
    }
    auto overlap = intersectSystemPresburgerCells(cell, point);
    if (!overlap)
      return overlap.takeError();
    return overlap->has_value();
  };
  for (const auto &clause : selection->clauses)
    for (const SystemPresburgerCell &cell : clause.cells) {
      auto intersects = intersectsPoint(cell);
      if (!intersects)
        return intersects.takeError();
      if (*intersects)
        selected.insert(clause.target);
    }
  if (selection->defaultPlanOrdinal)
    for (const SystemPresburgerCell &cell : defaultCells) {
      auto intersects = intersectsPoint(cell);
      if (!intersects)
        return intersects.takeError();
      if (*intersects)
        selected.insert(*selection->defaultPlanOrdinal);
    }
  if (selected.empty())
    return invalid("logical point selects no service plan");
  if (selected.size() != 1)
    return invalid("logical point has an ambiguous service plan across legal "
                   "symbol valuations");
  const std::uint64_t ordinal = *selected.begin();
  if (!llvm::any_of(realization.plans,
                    [&](const auto &plan) { return plan.ordinal == ordinal; }))
    return invalid("selected service plan is absent");
  return ordinal;
}

llvm::Expected<std::uint64_t> selectSystemDynamicWorkServicePlanOrdinal(
    const SystemServiceRealizationView &realization,
    const ServicePlanSelectionAnchor &anchor,
    const ExecutionContextKey &context,
    ::dataflow::DynamicWorkStableItemKey stableItem) {
  const SystemServicePlanSelectionView *selection = nullptr;
  for (const SystemServicePlanSelectionView &candidate :
       realization.selections) {
    if (!(candidate.key.anchor == anchor) ||
        !(candidate.key.context == context))
      continue;
    if (selection)
      return invalid("service plan selection key is duplicated");
    selection = &candidate;
  }
  if (!selection)
    return invalid("service plan selection key is absent");
  if (selection->relationKind !=
      ::mapping::SystemBindingRelationKind::StableKeyLookup)
    return invalid("service plan selection is not a stable-key lookup");
  std::optional<std::uint64_t> selected;
  for (const auto &entry : selection->stableKeyEntries) {
    if (!(entry.key == stableItem))
      continue;
    if (selected && *selected != entry.target)
      return invalid("stable item selects multiple service plans");
    selected = entry.target;
  }
  if (!selected)
    return invalid("stable item selects no service plan");
  if (!llvm::any_of(realization.plans, [&](const auto &plan) {
        return plan.ordinal == *selected;
      }))
    return invalid("selected service plan is absent");
  return *selected;
}

} // namespace loom::mapping
