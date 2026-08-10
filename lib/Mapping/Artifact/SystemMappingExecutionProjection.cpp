#include "Mapping/Artifact/SystemMappingExecutionProjection.h"

#include "Common/ArtifactLocalReference.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"

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
  if (logical->kind == ::dataflow::ThreadDomainKind::DynamicWork)
    return invalid("dynamic root domain has no Presburger projection");
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
        *key, ConcreteRelationPart<Target>{clause.target, {}});
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
        *key, ConcreteRelationPart<Target>{*defaultTarget, {}});
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

template <typename Target> std::string fabricKey(const Target &target) {
  const auto bytes = ::loom::fabric::canonicalFabricBytes(target);
  return byteKey(bytes);
}

llvm::Expected<std::string> mappingKey(const ArtifactRootReference &reference) {
  return byteKey(encodeArtifactRootReference(reference));
}

struct ThreadProjection final {
  ::dataflow::RootThreadLaunchRef root;
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
    auto parts = concretizeRelation(
        binding.clauses, binding.defaultTarget, *domain,
        [](const auto &target) -> llvm::Expected<std::string> {
          return fabricKey(target);
        });
    if (!parts)
      return parts.takeError();
    for (const auto &part : *parts)
      result.instructionDomains.push_back(SystemInstructionContextDomain{
          binding.key, InstructionExecutionContextKey{part.target},
          part.cells});
    if (!threads
             .emplace(*rootKey,
                      ThreadProjection{binding.key, std::move(*parts)})
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
    auto graphParts = concretizeRelation(
        binding.clauses, binding.defaultTarget, *domain,
        [](const auto &target) { return mappingKey(target); });
    if (!graphParts)
      return graphParts.takeError();

    std::map<std::string, SystemSpatialContextDomain> grouped;
    for (const auto &graphPart : *graphParts)
      for (const auto &threadPart : thread->second.parts)
        for (const auto &graphCell : graphPart.cells)
          for (const auto &threadCell : threadPart.cells) {
            auto overlap =
                intersectSystemPresburgerCells(graphCell, threadCell);
            if (!overlap)
              return overlap.takeError();
            if (!*overlap)
              continue;
            SpatialExecutionContextKey context{threadPart.target,
                                               graphPart.target.artifact};
            auto encoded = encodeExecutionContextKey(context);
            if (!encoded)
              return encoded.takeError();
            auto [entry, inserted] = grouped.try_emplace(
                byteKey(*encoded),
                SystemSpatialContextDomain{
                    binding.key, graphPart.target, context, {}});
            entry->second.cells.push_back(std::move(**overlap));
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

} // namespace loom::mapping
