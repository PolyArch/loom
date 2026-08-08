#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "MappingAssemblyInternal.h"
#include "SystemMappingClosure.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactLocalReference.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "Mapping/IR/MappingDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::mapping {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_mapping_execution_invalid: " +
                                     message);
}

std::vector<std::uint8_t> unsignedBytes(mlir::DenseI8ArrayAttr record) {
  std::vector<std::uint8_t> result;
  result.reserve(record.size());
  for (std::int8_t byte : record.asArrayRef())
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

std::string byteKey(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

template <typename Ref, typename Attr>
llvm::Expected<Ref> decodeDataflow(Attr attribute,
                                   const ArtifactIdentity &owner) {
  return ::dataflow::decodeDataflowReference<Ref>(
      unsignedBytes(attribute.getRecord()), owner);
}

template <typename Ref, typename Attr>
llvm::Expected<Ref> decodeFabric(Attr attribute) {
  return ::loom::fabric::decodeFabricRef<Ref>(
      unsignedBytes(attribute.getRecord()));
}

llvm::Expected<ArtifactRootReference>
decodeRootReference(::mapping::ArtifactRootReferenceAttr attribute) {
  std::vector<std::uint8_t> bytes = unsignedBytes(attribute.getRecord());
  auto decoded = decodeArtifactRootReferencePrefix(bytes);
  if (!decoded)
    return decoded.takeError();
  if (decoded->byteCount != bytes.size() ||
      encodeArtifactRootReference(decoded->reference) != bytes)
    return invalid("SpatialMapping import reference is not canonical");
  return std::move(decoded->reference);
}

struct ParsedSystemRoot final {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  ::mapping::SystemOp root;
};

llvm::Expected<ParsedSystemRoot>
parseSystemRoot(const CanonicalSemanticBytes &bytes) {
  std::string wrapped = "module {\n";
  wrapped.append(reinterpret_cast<const char *>(bytes.bytes().data()),
                 bytes.bytes().size());
  wrapped += "}\n";
  mlir::DialectRegistry registry;
  registry.insert<::mapping::MappingDialect>();
  auto context = std::make_unique<mlir::MLIRContext>(
      registry, mlir::MLIRContext::Threading::DISABLED);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(wrapped, context.get());
  if (!module)
    return invalid("canonical SystemMapping payload cannot be parsed");
  ::mapping::SystemOp root;
  unsigned count = 0;
  for (mlir::Operation &operation : module->getBody()->without_terminator()) {
    auto candidate = mlir::dyn_cast<::mapping::SystemOp>(operation);
    if (!candidate)
      return invalid("mapping artifact contains a non-SystemMapping root");
    root = candidate;
    ++count;
  }
  if (count != 1)
    return invalid("payload must contain exactly one SystemMapping root");
  if (mlir::failed(mlir::verify(root)))
    return invalid("SystemMapping root is structurally invalid");
  return ParsedSystemRoot{std::move(context), std::move(module), root};
}

llvm::Expected<SystemPresburgerCell>
decodeCell(::mapping::SystemPresburgerCellAttr attribute) {
  SystemPresburgerCell cell;
  cell.dimensionCount = attribute.getDimensionCount();
  cell.symbolCount = attribute.getSymbolCount();
  cell.localCount = attribute.getLocalCount();
  const auto appendRows = [](mlir::ArrayAttr attributes,
                             std::vector<std::vector<std::int64_t>> &rows) {
    rows.reserve(attributes.size());
    for (mlir::Attribute attribute : attributes) {
      auto values = mlir::cast<mlir::DenseI64ArrayAttr>(attribute).asArrayRef();
      rows.emplace_back(values.begin(), values.end());
    }
  };
  appendRows(attribute.getEqualities(), cell.equalities);
  appendRows(attribute.getInequalities(), cell.inequalities);
  return canonicalizeSystemPresburgerCell(cell);
}

llvm::Expected<SystemPresburgerCell>
legalDomain(const ::dataflow::CanonicalRootThreadLogicalDomainView &domain) {
  if (domain.kind == ::dataflow::ThreadDomainKind::DynamicWork)
    return invalid(
        "StableKeyLookup is unavailable without a Dataflow stable-key owner");
  SystemPresburgerCell cell;
  cell.dimensionCount = domain.coordinateRank;
  cell.symbolCount = static_cast<std::uint32_t>(domain.launchParameters.size());
  const std::size_t width = static_cast<std::size_t>(cell.dimensionCount) +
                            cell.symbolCount + cell.localCount + 1;
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

struct ExpectedBinding final {
  std::variant<::dataflow::RootThreadLaunchRef,
               ::dataflow::RootedGraphLaunchRef>
      key;
  SystemPresburgerCell legalDomain;
  std::vector<std::uint8_t> canonicalKey;
};

llvm::Expected<std::vector<ExpectedBinding>> collectExpectedBindings(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots) {
  std::vector<ExpectedBinding> result;
  std::map<std::uint64_t, SystemPresburgerCell> domains;
  for (const auto &root : roots) {
    auto logical = dataflow.projectRootThreadLogicalDomain(root);
    if (!logical)
      return logical.takeError();
    auto domain = legalDomain(*logical);
    if (!domain)
      return domain.takeError();
    auto key = ::dataflow::encodeDataflowReference(dataflow.identity(), root);
    if (!key)
      return key.takeError();
    domains.emplace(root.entity.value(), *domain);
    result.push_back({root, std::move(*domain), std::move(*key)});
  }
  llvm::Error callbackError = llvm::Error::success();
  dataflow.forEachRootedGraphLaunch(
      [&](::dataflow::RootedGraphLaunchRef graph) {
        if (callbackError)
          return;
        auto found = domains.find(graph.rootThreadLaunch.entity.value());
        if (found == domains.end())
          return;
        auto whole = dataflow.projectWholeRootedGraphLogicalDomain(graph);
        if (!whole) {
          callbackError = whole.takeError();
          return;
        }
        if (!*whole) {
          callbackError = invalid(
              "exact rooted graph may-domain projection is unavailable");
          return;
        }
        auto key =
            ::dataflow::encodeDataflowReference(dataflow.identity(), graph);
        if (!key) {
          callbackError = key.takeError();
          return;
        }
        result.push_back({graph, found->second, std::move(*key)});
      });
  if (callbackError)
    return std::move(callbackError);
  llvm::sort(result, [](const auto &lhs, const auto &rhs) {
    const std::uint32_t lhsKind =
        std::holds_alternative<::dataflow::RootThreadLaunchRef>(lhs.key) ? 0
                                                                         : 1;
    const std::uint32_t rhsKind =
        std::holds_alternative<::dataflow::RootThreadLaunchRef>(rhs.key) ? 0
                                                                         : 1;
    return std::tie(lhsKind, lhs.canonicalKey) <
           std::tie(rhsKind, rhs.canonicalKey);
  });
  return result;
}

llvm::Error validateRelation(llvm::ArrayRef<SystemPresburgerCell> cells,
                             bool hasDefault,
                             const SystemPresburgerCell &domain) {
  auto analysis = analyzeSystemPresburgerPartition(cells, domain);
  if (!analysis)
    return analysis.takeError();
  if (!analysis->liesWithinLegalDomain)
    return invalid("binding cell extends beyond its Dataflow may-domain");
  if (!analysis->cellsAreDisjoint)
    return invalid("binding relation has overlapping Presburger cells");
  if (hasDefault && analysis->coversLegalDomain)
    return invalid("binding default is forbidden for an empty complement");
  if (!hasDefault && !analysis->coversLegalDomain)
    return invalid("binding relation does not cover its Dataflow may-domain");
  return llvm::Error::success();
}

std::string targetClassKey(const ArtifactIdentity &module,
                           ::loom::fabric::FabricModuleTemplateRef target) {
  std::string key = byteKey(module.bytes());
  auto targetBytes = ::loom::fabric::canonicalFabricBytes(target);
  key.append(reinterpret_cast<const char *>(targetBytes.data()),
             targetBytes.size());
  return key;
}

llvm::Expected<std::string>
coreTargetClass(const ::loom::fabric::FabricSystemRootView &fabric,
                ::loom::fabric::AccCoreOccurrenceRef core) {
  if (!llvm::is_contained(fabric.artifact().accCoreOccurrences(), core))
    return invalid("ThreadExecutionBinding names an AccCore outside F");
  auto target = fabric.spatialCoreTarget(core);
  if (!target ||
      target->dependencyOrdinal >= fabric.artifact().importedModules().size())
    return invalid("selected AccCore has no exact SpatialCore target");
  const auto &module =
      fabric.artifact().importedModules()[target->dependencyOrdinal];
  auto root = module.moduleRootTemplate();
  if (!root || *root != target->target)
    return invalid("selected AccCore SpatialCore target is inconsistent");
  return targetClassKey(module.identity(), *root);
}

llvm::Expected<std::string>
mappingTargetClass(const ::loom::fabric::FabricSystemRootView &fabric,
                   const SpatialMappingView &mapping) {
  for (const auto &module : fabric.artifact().importedModules()) {
    if (module.identity() != mapping.fabricIdentity())
      continue;
    auto root = module.moduleRootTemplate();
    if (!root)
      return invalid("SpatialMapping import does not target a Module root");
    return targetClassKey(module.identity(), *root);
  }
  return invalid("SpatialMapping target Module is not imported by System");
}

template <typename Binding>
std::vector<SystemPresburgerCell> explicitCells(const Binding &binding) {
  std::vector<SystemPresburgerCell> cells;
  for (const auto &clause : binding.clauses)
    cells.insert(cells.end(), clause.cells.begin(), clause.cells.end());
  return cells;
}

llvm::Error verifyTargetCompatibility(
    const SystemThreadExecutionBindingView &thread,
    const SystemGraphExecutionBindingView &graph,
    const SystemPresburgerCell &domain,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const std::map<ArtifactRootReference, SpatialMappingView,
                   decltype(&artifactRootReferenceLess)> &mappings) {
  const auto threadCells = explicitCells(thread);
  const auto graphCells = explicitCells(graph);
  for (const auto &graphClause : graph.clauses) {
    auto mapping = mappings.find(graphClause.target);
    if (mapping == mappings.end())
      return invalid("graph clause target was not strictly imported");
    auto graphClass = mappingTargetClass(fabric, mapping->second);
    if (!graphClass)
      return graphClass.takeError();
    for (const auto &threadClause : thread.clauses) {
      bool intersects = false;
      for (const auto &graphCell : graphClause.cells)
        for (const auto &threadCell : threadClause.cells) {
          auto overlap = systemPresburgerCellsIntersect(graphCell, threadCell);
          if (!overlap)
            return overlap.takeError();
          intersects |= *overlap;
        }
      if (!intersects)
        continue;
      auto threadClass = coreTargetClass(fabric, threadClause.target);
      if (!threadClass)
        return threadClass.takeError();
      if (*threadClass != *graphClass)
        return invalid("graph and thread targets are incompatible");
    }
    if (thread.defaultTarget) {
      auto insideExplicit =
          systemPresburgerSetIsSubsetOf(graphClause.cells, threadCells);
      if (!insideExplicit)
        return insideExplicit.takeError();
      if (!*insideExplicit) {
        auto threadClass = coreTargetClass(fabric, *thread.defaultTarget);
        if (!threadClass)
          return threadClass.takeError();
        if (*threadClass != *graphClass)
          return invalid("graph target is incompatible with thread default");
      }
    }
  }
  if (graph.defaultTarget) {
    auto mapping = mappings.find(*graph.defaultTarget);
    if (mapping == mappings.end())
      return invalid("graph default target was not strictly imported");
    auto graphClass = mappingTargetClass(fabric, mapping->second);
    if (!graphClass)
      return graphClass.takeError();
    for (const auto &threadClause : thread.clauses) {
      auto insideExplicit =
          systemPresburgerSetIsSubsetOf(threadClause.cells, graphCells);
      if (!insideExplicit)
        return insideExplicit.takeError();
      if (!*insideExplicit) {
        auto threadClass = coreTargetClass(fabric, threadClause.target);
        if (!threadClass)
          return threadClass.takeError();
        if (*threadClass != *graphClass)
          return invalid("graph default is incompatible with thread target");
      }
    }
    if (thread.defaultTarget) {
      std::vector<SystemPresburgerCell> explicitUnion = threadCells;
      explicitUnion.insert(explicitUnion.end(), graphCells.begin(),
                           graphCells.end());
      auto covered = systemPresburgerSetIsSubsetOf({domain}, explicitUnion);
      if (!covered)
        return covered.takeError();
      if (!*covered) {
        auto threadClass = coreTargetClass(fabric, *thread.defaultTarget);
        if (!threadClass)
          return threadClass.takeError();
        if (*threadClass != *graphClass)
          return invalid("graph and thread defaults are incompatible");
      }
    }
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<SystemExecutionBindingView> strictImportSystemExecutionBindings(
    const CanonicalSemanticBytes &bytes,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ArtifactStore &store) {
  auto parsed = parseSystemRoot(bytes);
  if (!parsed)
    return parsed.takeError();
  auto dataflowIdentity = ArtifactIdentity::fromBytes(
      unsignedBytes(parsed->root.getDataflow().getRecord()));
  auto fabricIdentity = ArtifactIdentity::fromBytes(
      unsignedBytes(parsed->root.getFabric().getRecord()));
  if (!dataflowIdentity)
    return dataflowIdentity.takeError();
  if (!fabricIdentity)
    return fabricIdentity.takeError();
  if (*dataflowIdentity != dataflow.identity() ||
      *fabricIdentity != fabric.artifact().identity())
    return invalid("SystemMapping root has foreign D/F owners");

  std::vector<::dataflow::RootThreadLaunchRef> roots;
  for (mlir::Attribute attribute : parsed->root.getRootThreadLaunches()) {
    auto root = decodeDataflow<::dataflow::RootThreadLaunchRef>(
        mlir::cast<::mapping::RootThreadLaunchRefAttr>(attribute),
        dataflow.identity());
    if (!root)
      return root.takeError();
    auto resolved = dataflow.resolve(*root);
    if (!resolved)
      return resolved.takeError();
    roots.push_back(*root);
  }
  auto expected = collectExpectedBindings(dataflow, roots);
  if (!expected)
    return expected.takeError();
  std::map<std::string, const ExpectedBinding *> expectedByKey;
  for (const ExpectedBinding &binding : *expected) {
    const std::uint32_t kind =
        std::holds_alternative<::dataflow::RootThreadLaunchRef>(binding.key)
            ? 0
            : 1;
    expectedByKey.emplace(std::to_string(kind) + byteKey(binding.canonicalKey),
                          &binding);
  }

  std::vector<ArtifactRootReference> imports;
  std::map<ArtifactRootReference, SpatialMappingView,
           decltype(&artifactRootReferenceLess)>
      importedMappings(&artifactRootReferenceLess);
  for (mlir::Attribute attribute : parsed->root.getSpatialMappingImports()) {
    auto reference = decodeRootReference(
        mlir::cast<::mapping::ArtifactRootReferenceAttr>(attribute));
    if (!reference)
      return reference.takeError();
    if (reference->schemaIdentity != mappingArtifactSchema.identity ||
        reference->schemaVersion != mappingArtifactSchema.version)
      return invalid("import table contains a non-Mapping reference");
    auto mapping = importSpatialMapping(*reference, store);
    if (!mapping)
      return mapping.takeError();
    if (mapping->view().dataflowIdentity() != dataflow.identity())
      return invalid("SpatialMapping import has a foreign Dataflow owner");
    if (!importedMappings.emplace(*reference, mapping->view()).second)
      return invalid("SpatialMapping import table contains a duplicate");
    if (auto target = mappingTargetClass(fabric, mapping->view()); !target)
      return target.takeError();
    imports.push_back(*reference);
  }

  std::vector<SystemThreadExecutionBindingView> threadBindings;
  std::vector<SystemGraphExecutionBindingView> graphBindings;
  std::map<std::string, std::size_t> threadByKey;
  std::set<std::string> seenKeys;
  for (mlir::Operation &operation : parsed->root.getBody().front()) {
    if (auto binding =
            mlir::dyn_cast<::mapping::ThreadExecutionBindingOp>(operation)) {
      auto key = decodeDataflow<::dataflow::RootThreadLaunchRef>(
          binding.getKey(), dataflow.identity());
      if (!key)
        return key.takeError();
      auto keyBytes =
          ::dataflow::encodeDataflowReference(dataflow.identity(), *key);
      if (!keyBytes)
        return keyBytes.takeError();
      const std::string lookup = "0" + byteKey(*keyBytes);
      auto expectedIt = expectedByKey.find(lookup);
      if (expectedIt == expectedByKey.end() || !seenKeys.insert(lookup).second)
        return invalid("ThreadExecutionBinding has a foreign or duplicate key");
      SystemThreadExecutionBindingView view{*key, {}, std::nullopt};
      std::vector<SystemPresburgerCell> cells;
      for (auto clause : binding.getBody()
                             .front()
                             .getOps<::mapping::ThreadPresburgerClauseOp>()) {
        auto target = decodeFabric<::loom::fabric::AccCoreOccurrenceRef>(
            clause.getTarget());
        if (!target)
          return target.takeError();
        if (auto targetClass = coreTargetClass(fabric, *target); !targetClass)
          return targetClass.takeError();
        SystemPresburgerClauseView<::loom::fabric::AccCoreOccurrenceRef>
            imported{{}, *target};
        for (mlir::Attribute raw : clause.getCells()) {
          auto cell =
              decodeCell(mlir::cast<::mapping::SystemPresburgerCellAttr>(raw));
          if (!cell)
            return cell.takeError();
          cells.push_back(*cell);
          imported.cells.push_back(std::move(*cell));
        }
        view.clauses.push_back(std::move(imported));
      }
      if (binding.getDefaultTarget()) {
        auto target = decodeFabric<::loom::fabric::AccCoreOccurrenceRef>(
            *binding.getDefaultTarget());
        if (!target)
          return target.takeError();
        if (auto targetClass = coreTargetClass(fabric, *target); !targetClass)
          return targetClass.takeError();
        view.defaultTarget = *target;
      }
      if (llvm::Error error =
              validateRelation(cells, view.defaultTarget.has_value(),
                               expectedIt->second->legalDomain))
        return std::move(error);
      threadByKey.emplace(byteKey(*keyBytes), threadBindings.size());
      threadBindings.push_back(std::move(view));
      continue;
    }
    if (mlir::isa<::mapping::ServiceRealizationOp, ::mapping::ResourceUseOp>(
            operation))
      continue;
    auto binding =
        mlir::dyn_cast<::mapping::GraphExecutionBindingOp>(operation);
    if (!binding)
      return invalid(
          "execution importer encountered a non-execution System record");
    auto key = decodeDataflow<::dataflow::RootedGraphLaunchRef>(
        binding.getKey(), dataflow.identity());
    if (!key)
      return key.takeError();
    auto keyBytes =
        ::dataflow::encodeDataflowReference(dataflow.identity(), *key);
    if (!keyBytes)
      return keyBytes.takeError();
    const std::string lookup = "1" + byteKey(*keyBytes);
    auto expectedIt = expectedByKey.find(lookup);
    if (expectedIt == expectedByKey.end() || !seenKeys.insert(lookup).second)
      return invalid("GraphExecutionBinding has a foreign or duplicate key");
    SystemGraphExecutionBindingView view{*key, {}, std::nullopt};
    std::vector<SystemPresburgerCell> cells;
    for (auto clause : binding.getBody()
                           .front()
                           .getOps<::mapping::GraphPresburgerClauseOp>()) {
      const std::uint64_t ordinal = clause.getTarget().getOrdinal();
      if (ordinal >= imports.size())
        return invalid("graph clause names an absent SpatialMapping import");
      SystemPresburgerClauseView<ArtifactRootReference> imported{
          {}, imports[ordinal]};
      for (mlir::Attribute raw : clause.getCells()) {
        auto cell =
            decodeCell(mlir::cast<::mapping::SystemPresburgerCellAttr>(raw));
        if (!cell)
          return cell.takeError();
        cells.push_back(*cell);
        imported.cells.push_back(std::move(*cell));
      }
      view.clauses.push_back(std::move(imported));
    }
    if (binding.getDefaultTarget()) {
      const std::uint64_t ordinal = binding.getDefaultTarget()->getOrdinal();
      if (ordinal >= imports.size())
        return invalid("graph default names an absent SpatialMapping import");
      view.defaultTarget = imports[ordinal];
    }
    if (llvm::Error error =
            validateRelation(cells, view.defaultTarget.has_value(),
                             expectedIt->second->legalDomain))
      return std::move(error);
    graphBindings.push_back(std::move(view));
  }
  if (seenKeys.size() != expected->size())
    return invalid("execution bindings do not cover the exact D closure");

  std::vector<ArtifactRootReference> selectedImports;
  for (const auto &binding : graphBindings) {
    for (const auto &clause : binding.clauses)
      selectedImports.push_back(clause.target);
    if (binding.defaultTarget)
      selectedImports.push_back(*binding.defaultTarget);
  }
  llvm::sort(selectedImports, artifactRootReferenceLess);
  selectedImports.erase(
      std::unique(selectedImports.begin(), selectedImports.end()),
      selectedImports.end());
  if (selectedImports != imports)
    return invalid("import table is not the exact selected B_graph range");

  for (const auto &graph : graphBindings) {
    auto parentKey = ::dataflow::encodeDataflowReference(
        dataflow.identity(), graph.key.rootThreadLaunch);
    if (!parentKey)
      return parentKey.takeError();
    auto parent = threadByKey.find(byteKey(*parentKey));
    if (parent == threadByKey.end())
      return invalid("graph binding has no parent thread binding");
    const ExpectedBinding *expectedGraph = nullptr;
    auto graphKey =
        ::dataflow::encodeDataflowReference(dataflow.identity(), graph.key);
    if (!graphKey)
      return graphKey.takeError();
    auto expectedIt = expectedByKey.find("1" + byteKey(*graphKey));
    if (expectedIt == expectedByKey.end())
      return invalid("graph binding lost its expected logical domain");
    expectedGraph = expectedIt->second;
    if (llvm::Error error = verifyTargetCompatibility(
            threadBindings[parent->second], graph, expectedGraph->legalDomain,
            fabric, importedMappings))
      return std::move(error);
  }

  auto canonical = writeCanonicalSystemMappingAssembly(parsed->root);
  if (!canonical)
    return canonical.takeError();
  if (!canonical->bytes().equals(bytes.bytes()))
    return invalid("stored SystemMapping execution payload is not canonical");
  return SystemExecutionBindingView(std::move(roots), std::move(imports),
                                    std::move(threadBindings),
                                    std::move(graphBindings));
}

llvm::Expected<SystemMappingView> importSystemMappingView(
    const ArtifactIdentity &mappingIdentity, ::mapping::SystemOp root,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ArtifactStore &store) {
  auto canonical = writeCanonicalSystemMappingAssembly(root);
  if (!canonical)
    return canonical.takeError();
  if (finalizeArtifactIdentity(mappingArtifactSchema, *canonical) !=
      mappingIdentity)
    return invalid("mapping identity does not match canonical bytes");
  auto execution =
      strictImportSystemExecutionBindings(*canonical, dataflow, fabric, store);
  if (!execution)
    return execution.takeError();
  auto closure =
      detail::importSystemMappingClosure(root, dataflow, fabric, *execution);
  if (!closure)
    return closure.takeError();
  return SystemMappingView(mappingIdentity, dataflow.identity(),
                           fabric.artifact().identity(), std::move(*execution),
                           std::move(closure->services),
                           std::move(closure->resourceUses));
}

llvm::Error verifySystemMappingBase(
    ::mapping::SystemOp source,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ArtifactStore &store) {
  auto assembly = detail::prepareCanonicalSystemMappingAssembly(source);
  if (!assembly)
    return assembly.takeError();
  const ArtifactIdentity identity =
      finalizeArtifactIdentity(mappingArtifactSchema, assembly->bytes);
  auto view = importSystemMappingView(
      identity, mlir::cast<::mapping::SystemOp>(assembly->root.get()), dataflow,
      fabric, store);
  if (!view)
    return view.takeError();
  return llvm::Error::success();
}

llvm::Expected<FinalizedSystemMapping>
finalizeSystemMapping(::mapping::SystemOp source,
                      const ::dataflow::CanonicalDataflowProgramView &dataflow,
                      const ::loom::fabric::FabricSystemRootView &fabric,
                      const SystemMappingConstraintSetView &constraints,
                      const ArtifactStore &store) {
  if (constraints.dataflowIdentity() != dataflow.identity() ||
      constraints.fabricIdentity() != fabric.artifact().identity())
    return invalid("System constraint owner tuple does not match D/F");

  const ArtifactRootReference upstream[] = {
      {::dataflow::canonicalDataflowSchema.identity.str(),
       ::dataflow::canonicalDataflowSchema.version, dataflow.identity()},
      {::loom::fabric::fabricArtifactSchema.identity.str(),
       ::loom::fabric::fabricArtifactSchema.version,
       fabric.artifact().identity()},
      {mappingConstraintSetSchema.identity.str(),
       mappingConstraintSetSchema.version, constraints.identity()}};
  for (const auto &reference : upstream) {
    auto bytes = store.get(reference);
    if (!bytes)
      return bytes.takeError();
  }

  auto assembly = detail::prepareCanonicalSystemMappingAssembly(source);
  if (!assembly)
    return assembly.takeError();
  const ArtifactIdentity identity =
      finalizeArtifactIdentity(mappingArtifactSchema, assembly->bytes);
  auto root = mlir::cast<::mapping::SystemOp>(assembly->root.get());
  auto view = importSystemMappingView(identity, root, dataflow, fabric, store);
  if (!view)
    return view.takeError();
  if (constraints.rootThreadLaunches() !=
      view->executionBindings().rootThreadLaunches())
    return invalid("System constraint root scope does not match Mapping");
  if (llvm::Error error =
          admitSystemMappingConstraints(dataflow, fabric, constraints, *view))
    return std::move(error);

  auto stored = store.put(mappingArtifactSchema, assembly->bytes);
  if (!stored)
    return stored.takeError();
  if (*stored != identity)
    return invalid("ArtifactStore returned a different Mapping identity");
  ArtifactRootReference reference{mappingArtifactSchema.identity.str(),
                                  mappingArtifactSchema.version, identity};
  return FinalizedSystemMapping(std::move(reference),
                                std::move(assembly->bytes), std::move(*view));
}

llvm::Expected<FinalizedSystemMapping>
importSystemMapping(const ArtifactRootReference &reference,
                    const ArtifactStore &store) {
  if (reference.schemaIdentity != mappingArtifactSchema.identity ||
      reference.schemaVersion != mappingArtifactSchema.version)
    return invalid("root reference has the wrong Mapping schema");
  auto canonical = store.get(reference);
  if (!canonical)
    return canonical.takeError();
  if (finalizeArtifactIdentity(mappingArtifactSchema, *canonical) !=
      reference.artifact)
    return invalid("mapping identity does not match canonical bytes");
  auto parsed = parseSystemRoot(*canonical);
  if (!parsed)
    return parsed.takeError();
  auto replay = writeCanonicalSystemMappingAssembly(parsed->root);
  if (!replay)
    return replay.takeError();
  if (replay->bytes() != canonical->bytes())
    return invalid("stored SystemMapping payload is not canonical");
  auto dataflowIdentity = ArtifactIdentity::fromBytes(
      unsignedBytes(parsed->root.getDataflow().getRecord()));
  auto fabricIdentity = ArtifactIdentity::fromBytes(
      unsignedBytes(parsed->root.getFabric().getRecord()));
  if (!dataflowIdentity)
    return dataflowIdentity.takeError();
  if (!fabricIdentity)
    return fabricIdentity.takeError();
  auto dataflow = ::dataflow::importCanonicalDataflow(
      {::dataflow::canonicalDataflowSchema.identity.str(),
       ::dataflow::canonicalDataflowSchema.version, *dataflowIdentity},
      store);
  if (!dataflow)
    return dataflow.takeError();
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return dataflowView.takeError();
  auto fabric = ::loom::fabric::importEntireFabricRoot(
      {::loom::fabric::fabricArtifactSchema.identity.str(),
       ::loom::fabric::fabricArtifactSchema.version, *fabricIdentity},
      store);
  if (!fabric)
    return fabric.takeError();
  auto system = ::loom::fabric::requireSystemRoot(fabric->view());
  if (!system)
    return system.takeError();
  auto view = importSystemMappingView(reference.artifact, parsed->root,
                                      *dataflowView, *system, store);
  if (!view)
    return view.takeError();
  return FinalizedSystemMapping(reference, std::move(*canonical),
                                std::move(*view));
}

} // namespace loom::mapping
