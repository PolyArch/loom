#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "MappingAssemblyInternal.h"
#include "SystemMappingCapacityVerification.h"
#include "SystemMappingClosure.h"
#include "SystemMappingHandshakeVerification.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/InvocationDiagnosticLog.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemReferenceRemapper.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingProgressAnalysis.h"
#include "Mapping/Artifact/SystemMappingClosureProjection.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "Mapping/IR/MappingDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <chrono>
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
namespace detail {

inline constexpr std::uint64_t systemMappingImportAlgorithmVersion = 2;

struct SystemMappingImportSessionKey final {
  ArtifactRootReference reference;
  std::uint64_t algorithmVersion = systemMappingImportAlgorithmVersion;

  friend bool operator==(const SystemMappingImportSessionKey &lhs,
                         const SystemMappingImportSessionKey &rhs) {
    return lhs.reference == rhs.reference &&
           lhs.algorithmVersion == rhs.algorithmVersion;
  }
};

struct SystemMappingImportSessionEntry final {
  SystemMappingImportSessionKey key;
  std::shared_ptr<const FinalizedSystemMapping> mapping;
  std::uint64_t retainedBytes = 0;
};

class SystemMappingImportSessionState final {
public:
  SystemMappingImportSessionState(const ArtifactStore &store,
                                  std::size_t entryLimit)
      : store_(&store), entryLimit_(entryLimit) {}

  bool owns(const ArtifactStore &store) const { return store_ == &store; }

  std::shared_ptr<const FinalizedSystemMapping>
  find(const ArtifactRootReference &reference) {
    ++statistics_.importRequests;
    ++statistics_.deterministicWork;
    const SystemMappingImportSessionKey key{reference};
    const auto found = llvm::find_if(
        entries_, [&](const auto &entry) { return entry.key == key; });
    if (found == entries_.end()) {
      ++statistics_.cacheMisses;
      return {};
    }
    ++statistics_.cacheHits;
    return found->mapping;
  }

  std::shared_ptr<const FinalizedSystemMapping>
  insert(const ArtifactRootReference &reference,
         std::shared_ptr<const FinalizedSystemMapping> mapping,
         std::uint64_t retainedBytes, std::uint64_t constructionNanoseconds,
         std::uint64_t deterministicWork) {
    ++statistics_.uniqueConstructions;
    statistics_.bytesRead += mapping->canonicalBytes().bytes().size();
    statistics_.constructionNanoseconds += constructionNanoseconds;
    statistics_.deterministicWork += deterministicWork;
    if (entries_.size() >= entryLimit_) {
      ++statistics_.uncachedConstructions;
      return mapping;
    }
    entries_.push_back({{reference}, mapping, retainedBytes});
    statistics_.retainedBytes += retainedBytes;
    statistics_.entryCount = entries_.size();
    return mapping;
  }

  SystemMappingImportSessionStatistics statistics() const {
    return statistics_;
  }

private:
  const ArtifactStore *store_ = nullptr;
  std::size_t entryLimit_ = 0;
  std::vector<SystemMappingImportSessionEntry> entries_;
  SystemMappingImportSessionStatistics statistics_;
};

} // namespace detail

char SystemMappingIncompleteError::ID;

SystemMappingIncompleteError::SystemMappingIncompleteError(
    SystemMappingIncompleteReason reason, std::string diagnostic)
    : reason_(reason), diagnostic_(std::move(diagnostic)) {}

void SystemMappingIncompleteError::log(llvm::raw_ostream &stream) const {
  stream << "system_mapping_incomplete: " << diagnostic_;
}

std::error_code SystemMappingIncompleteError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

char SystemMappingRejectedError::ID;

SystemMappingRejectedError::SystemMappingRejectedError(
    SystemMappingClosureFindingKind finding, std::string diagnostic)
    : finding_(finding), diagnostic_(std::move(diagnostic)) {}

void SystemMappingRejectedError::log(llvm::raw_ostream &stream) const {
  stream << "system_mapping_rejected: " << diagnostic_;
}

std::error_code SystemMappingRejectedError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

const SystemMappingClosureProjection &
FinalizedSystemMapping::verifiedClosure() const {
  return *verifiedClosure_;
}

namespace {

using MonotonicClock = std::chrono::steady_clock;

thread_local detail::SystemMappingImportSessionState
    *currentSystemMappingImportSession = nullptr;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_mapping_execution_invalid: " +
                                     message);
}

llvm::Error cancelled() {
  return llvm::make_error<SystemMappingIncompleteError>(
      SystemMappingIncompleteReason::CancelledOrTimeout,
      "System Mapping finalization was cancelled or timed out");
}

std::uint64_t
retainedSystemMappingBytes(const FinalizedSystemMapping &mapping) {
  const SystemMappingView &view = mapping.view();
  const SystemMappingClosureProjection &closure = mapping.verifiedClosure();
  std::uint64_t closureBytes =
      sizeof(closure) +
      closure.executionContexts.instructionDomains.capacity() *
          sizeof(SystemInstructionContextDomain) +
      closure.executionContexts.spatialDomains.capacity() *
          sizeof(SystemSpatialContextDomain) +
      closure.serviceRealizations.capacity() *
          sizeof(SystemServiceRealizationView) +
      closure.progressBasis.residualCycle.capacity() *
          sizeof(::dataflow::ActorRef) +
      closure.routeObligations.capacity() *
          sizeof(MappingRouteProgressObligationProjection) +
      closure.capacityCells.capacity() * sizeof(SystemCapacityCellProjection) +
      closure.resourceActivations.capacity() *
          sizeof(SystemResourceActivationProjection);
  for (const auto &domain : closure.executionContexts.instructionDomains)
    closureBytes += domain.cells.capacity() * sizeof(SystemPresburgerCell);
  for (const auto &domain : closure.executionContexts.spatialDomains)
    closureBytes += domain.cells.capacity() * sizeof(SystemPresburgerCell);
  for (const auto &activation : closure.resourceActivations) {
    closureBytes +=
        activation.relationDomain.capacity() * sizeof(SystemPresburgerCell) +
        activation.triggerAlternatives.capacity() *
            sizeof(::dataflow::EventFamilyKey) +
        activation.parameters.capacity() * sizeof(::fabric::UsePatternValue) +
        activation.sharingAssignments.capacity() *
            sizeof(::fabric::UsePatternValue) +
        activation.capacityClaims.capacity() *
            sizeof(SystemCapacityClaimProjection) +
        activation.causalRelease.capacity() *
            sizeof(SystemCausalReleasePointProjection);
    for (const auto &release : activation.causalRelease)
      closureBytes +=
          release.alternatives.capacity() * sizeof(::dataflow::EventFamilyKey) +
          (release.guaranteedOffset ? release.guaranteedOffset->capacity() : 0);
  }
  return sizeof(mapping) + mapping.canonicalBytes().bytes().size() +
         view.executionBindings().spatialMappingImports().size() *
             sizeof(ArtifactRootReference) +
         view.executionBindings().threadBindings().size() *
             sizeof(SystemThreadExecutionBindingView) +
         view.executionBindings().graphBindings().size() *
             sizeof(SystemGraphExecutionBindingView) +
         view.serviceRealizations().size() *
             sizeof(SystemServiceRealizationView) +
         view.resourceUses().size() * sizeof(SystemResourceUseView) +
         closureBytes;
}

std::uint64_t
deterministicSystemMappingWork(const FinalizedSystemMapping &mapping) {
  const SystemMappingView &view = mapping.view();
  const SystemMappingClosureProjection &closure = mapping.verifiedClosure();
  std::uint64_t closureWork =
      1 + closure.executionContexts.instructionDomains.size() +
      closure.executionContexts.spatialDomains.size() +
      closure.serviceRealizations.size() +
      closure.progressBasis.residualCycle.size() +
      closure.routeObligations.size() + closure.capacityCells.size() +
      closure.resourceActivations.size();
  for (const auto &domain : closure.executionContexts.instructionDomains)
    closureWork += domain.cells.size();
  for (const auto &domain : closure.executionContexts.spatialDomains)
    closureWork += domain.cells.size();
  for (const auto &activation : closure.resourceActivations) {
    closureWork +=
        activation.relationDomain.size() +
        activation.triggerAlternatives.size() + activation.parameters.size() +
        activation.sharingAssignments.size() +
        activation.capacityClaims.size() + activation.causalRelease.size();
    for (const auto &release : activation.causalRelease)
      closureWork +=
          release.alternatives.size() +
          (release.guaranteedOffset ? release.guaranteedOffset->size() : 0);
  }
  return 1 + view.executionBindings().spatialMappingImports().size() +
         view.executionBindings().threadBindings().size() +
         view.executionBindings().graphBindings().size() +
         view.serviceRealizations().size() + view.resourceUses().size() +
         closureWork;
}

llvm::StringRef spelling(SystemMappingImportVerificationDomain domain) {
  switch (domain) {
  case SystemMappingImportVerificationDomain::SourceInvocation:
    return "source_invocation";
  case SystemMappingImportVerificationDomain::IndependentReplay:
    return "independent_replay";
  }
  llvm_unreachable("unknown SystemMapping import verification domain");
}

std::vector<std::uint8_t> unsignedBytes(mlir::DenseI8ArrayAttr record) {
  std::vector<std::uint8_t> result;
  result.reserve(record.size());
  for (std::int8_t byte : record.asArrayRef())
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

std::vector<std::uint8_t> unsignedBytes(llvm::ArrayRef<std::int8_t> record) {
  std::vector<std::uint8_t> result;
  result.reserve(record.size());
  for (std::int8_t byte : record)
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

mlir::DenseI8ArrayAttr denseBytes(mlir::MLIRContext *context,
                                  llvm::ArrayRef<std::uint8_t> value) {
  llvm::SmallVector<std::int8_t, 32> bytes;
  bytes.reserve(value.size());
  for (std::uint8_t byte : value)
    bytes.push_back(static_cast<std::int8_t>(byte));
  return mlir::DenseI8ArrayAttr::get(context, bytes);
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

struct ExpectedLogicalDomain final {
  SystemPresburgerCell presburgerSurrogate;
  ::mapping::SystemBindingRelationKind relationKind =
      ::mapping::SystemBindingRelationKind::PresburgerPartition;
  std::vector<::dataflow::DynamicWorkStableItemKey> stableItemKeys;
};

llvm::Expected<ExpectedLogicalDomain>
legalDomain(const ::dataflow::CanonicalDataflowProgramView &dataflow,
            const ::dataflow::CanonicalRootThreadLogicalDomainView &domain) {
  if (domain.kind == ::dataflow::ThreadDomainKind::DynamicWork) {
    auto projection = dataflow.projectDynamicWork(domain.launch);
    if (!projection)
      return projection.takeError();
    if (projection->stableItemKeys.size() != 1)
      return invalid("DynamicWork stable-key domain is not singleton");
    auto cell = canonicalizeSystemPresburgerCell(SystemPresburgerCell{});
    if (!cell)
      return cell.takeError();
    return ExpectedLogicalDomain{
        std::move(*cell), ::mapping::SystemBindingRelationKind::StableKeyLookup,
        std::move(projection->stableItemKeys)};
  }
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
  auto canonical = canonicalizeSystemPresburgerCell(cell);
  if (!canonical)
    return canonical.takeError();
  return ExpectedLogicalDomain{
      std::move(*canonical),
      ::mapping::SystemBindingRelationKind::PresburgerPartition,
      {}};
}

struct ExpectedBinding final {
  std::variant<::dataflow::RootThreadLaunchRef,
               ::dataflow::RootedGraphLaunchRef>
      key;
  SystemPresburgerCell legalDomain;
  std::vector<std::uint8_t> canonicalKey;
  ::mapping::SystemBindingRelationKind relationKind =
      ::mapping::SystemBindingRelationKind::PresburgerPartition;
  std::vector<::dataflow::DynamicWorkStableItemKey> stableItemKeys;
};

llvm::Expected<std::vector<ExpectedBinding>> collectExpectedBindings(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots) {
  std::vector<ExpectedBinding> result;
  std::map<std::uint64_t, ExpectedLogicalDomain> domains;
  for (const auto &root : roots) {
    auto logical = dataflow.projectRootThreadLogicalDomain(root);
    if (!logical)
      return logical.takeError();
    auto domain = legalDomain(dataflow, *logical);
    if (!domain)
      return domain.takeError();
    auto key = ::dataflow::encodeDataflowReference(dataflow.identity(), root);
    if (!key)
      return key.takeError();
    domains.emplace(root.entity.value(), *domain);
    result.push_back({root, domain->presburgerSurrogate, std::move(*key),
                      domain->relationKind, domain->stableItemKeys});
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
        result.push_back({graph, found->second.presburgerSurrogate,
                          std::move(*key), found->second.relationKind,
                          found->second.stableItemKeys});
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

template <typename Target>
llvm::Error validateStableRelation(
    const std::vector<SystemStableKeyEntryView<Target>> &entries,
    llvm::ArrayRef<::dataflow::DynamicWorkStableItemKey> expectedKeys) {
  std::vector<std::vector<std::uint8_t>> actual;
  actual.reserve(entries.size());
  for (const auto &entry : entries)
    actual.push_back(::dataflow::encodeDynamicWorkStableItemKey(entry.key));
  std::vector<std::vector<std::uint8_t>> expected;
  expected.reserve(expectedKeys.size());
  for (const auto &key : expectedKeys)
    expected.push_back(::dataflow::encodeDynamicWorkStableItemKey(key));
  llvm::sort(actual);
  llvm::sort(expected);
  if (std::adjacent_find(actual.begin(), actual.end()) != actual.end())
    return invalid("StableKeyLookup contains a duplicate key");
  if (actual != expected)
    return invalid("StableKeyLookup does not cover its exact Dataflow key set");
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
  if (thread.relationKind != graph.relationKind)
    return invalid("graph and thread binding relation kinds differ");
  if (thread.relationKind ==
      ::mapping::SystemBindingRelationKind::StableKeyLookup) {
    std::map<std::vector<std::uint8_t>, ::loom::fabric::AccCoreOccurrenceRef>
        threadTargets;
    for (const auto &entry : thread.stableKeyEntries)
      threadTargets.emplace(
          ::dataflow::encodeDynamicWorkStableItemKey(entry.key), entry.target);
    for (const auto &entry : graph.stableKeyEntries) {
      auto threadTarget = threadTargets.find(
          ::dataflow::encodeDynamicWorkStableItemKey(entry.key));
      if (threadTarget == threadTargets.end())
        return invalid("graph stable key has no parent thread target");
      auto mapping = mappings.find(entry.target);
      if (mapping == mappings.end())
        return invalid("graph stable-key target was not strictly imported");
      auto graphClass = mappingTargetClass(fabric, mapping->second);
      if (!graphClass)
        return graphClass.takeError();
      auto threadClass = coreTargetClass(fabric, threadTarget->second);
      if (!threadClass)
        return threadClass.takeError();
      if (*threadClass != *graphClass)
        return invalid("graph and thread stable-key targets are incompatible");
    }
    if (threadTargets.size() != graph.stableKeyEntries.size())
      return invalid("graph stable-key domain differs from its parent thread");
    return llvm::Error::success();
  }

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
          if (graphCell.dimensionCount != threadCell.dimensionCount ||
              graphCell.symbolCount != threadCell.symbolCount)
            return invalid(
                "graph/thread compatibility crosses Presburger spaces for "
                "thread root " +
                llvm::Twine(thread.key.entity.value()) + ", graph root " +
                llvm::Twine(graph.key.rootThreadLaunch.entity.value()) +
                ", graph launch " +
                llvm::Twine(graph.key.staticGraphLaunch.entity.value()) +
                ": thread (dims=" + llvm::Twine(threadCell.dimensionCount) +
                ", symbols=" + llvm::Twine(threadCell.symbolCount) +
                "), graph (dims=" + llvm::Twine(graphCell.dimensionCount) +
                ", symbols=" + llvm::Twine(graphCell.symbolCount) + ")");
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

llvm::Expected<std::vector<ArtifactRootReference>>
decodeSpatialMappingImports(::mapping::SystemOp root) {
  std::vector<ArtifactRootReference> imports;
  imports.reserve(root.getSpatialMappingImports().size());
  for (mlir::Attribute attribute : root.getSpatialMappingImports()) {
    auto reference = decodeRootReference(
        mlir::cast<::mapping::ArtifactRootReferenceAttr>(attribute));
    if (!reference)
      return reference.takeError();
    if (reference->schemaIdentity != mappingArtifactSchema.identity ||
        reference->schemaVersion != mappingArtifactSchema.version)
      return invalid("import table contains a non-Mapping reference");
    imports.push_back(std::move(*reference));
  }
  return imports;
}

llvm::Error validateSpatialMappingImportContext(
    llvm::ArrayRef<ArtifactRootReference> imports,
    const SpatialMappingImportContext &context) {
  std::vector<ArtifactRootReference> canonical(imports.begin(), imports.end());
  llvm::sort(canonical, artifactRootReferenceLess);
  if (std::adjacent_find(canonical.begin(), canonical.end()) != canonical.end())
    return invalid("SpatialMapping import table contains a duplicate");
  for (const ArtifactRootReference &reference : canonical)
    if (!context.find(reference))
      return invalid("SpatialMapping import context does not cover the exact "
                     "SystemMapping import table");
  return llvm::Error::success();
}

} // namespace

SystemMappingImportSession::SystemMappingImportSession(
    const ArtifactStore &store, std::size_t entryLimit,
    SystemMappingImportSessionMode mode)
    : previous_(currentSystemMappingImportSession) {
  if (mode == SystemMappingImportSessionMode::ReuseEnclosing && previous_ &&
      previous_->owns(store)) {
    active_ = previous_;
  } else {
    state_ = std::make_unique<detail::SystemMappingImportSessionState>(
        store, entryLimit);
    active_ = state_.get();
  }
  currentSystemMappingImportSession = active_;
}

SystemMappingImportSession::~SystemMappingImportSession() {
  currentSystemMappingImportSession = previous_;
}

SystemMappingImportSessionStatistics
SystemMappingImportSession::statistics() const {
  return active_ ? active_->statistics()
                 : SystemMappingImportSessionStatistics{};
}

void emitSystemMappingImportSessionStatistics(
    SystemMappingImportVerificationDomain domain,
    const SystemMappingImportSessionStatistics &statistics) {
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary, InvocationDiagnosticStage::SystemPnr,
      InvocationDiagnosticEvent::SystemMappingImportSession, [&] {
        llvm::json::Object payload;
        payload["verification_domain"] = spelling(domain);
        payload["import_requests"] = statistics.importRequests;
        payload["cache_hits"] = statistics.cacheHits;
        payload["cache_misses"] = statistics.cacheMisses;
        payload["unique_constructions"] = statistics.uniqueConstructions;
        payload["uncached_constructions"] = statistics.uncachedConstructions;
        payload["bytes_read"] = statistics.bytesRead;
        payload["construction_time_ns"] = statistics.constructionNanoseconds;
        payload["deterministic_work"] = statistics.deterministicWork;
        payload["retained_bytes"] = statistics.retainedBytes;
        payload["entry_count"] = statistics.entryCount;
        return llvm::json::Value(std::move(payload));
      });
}

llvm::Expected<SystemExecutionBindingView> strictImportSystemExecutionBindings(
    const CanonicalSemanticBytes &bytes,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ArtifactStore &store,
    const SpatialMappingImportContext *spatialMappings,
    ExecutionControlView executionControl) {
  const auto interrupted = [&]() -> llvm::Error {
    return llvm::createStringError(std::errc::timed_out,
                                   "System execution import was interrupted");
  };
  if (executionControl.stopRequested())
    return interrupted();
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
    if (executionControl.stopRequested())
      return interrupted();
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

  auto decodedImports = decodeSpatialMappingImports(parsed->root);
  if (!decodedImports)
    return decodedImports.takeError();
  std::vector<ArtifactRootReference> imports = std::move(*decodedImports);
  std::optional<SpatialMappingImportContext> ownedSpatialMappings;
  if (!spatialMappings) {
    auto built = buildSpatialMappingImportContext(imports, store);
    if (!built)
      return built.takeError();
    ownedSpatialMappings.emplace(std::move(*built));
    spatialMappings = &*ownedSpatialMappings;
  }
  if (llvm::Error error =
          validateSpatialMappingImportContext(imports, *spatialMappings))
    return std::move(error);

  std::map<ArtifactRootReference, SpatialMappingView,
           decltype(&artifactRootReferenceLess)>
      importedMappings(&artifactRootReferenceLess);
  for (const ArtifactRootReference &reference : imports) {
    if (executionControl.stopRequested())
      return interrupted();
    auto mapping = resolveSpatialMappingImport(*spatialMappings, reference);
    if (!mapping)
      return mapping.takeError();
    if ((*mapping)->view().dataflowIdentity() != dataflow.identity())
      return invalid("SpatialMapping import has a foreign Dataflow owner");
    if (!importedMappings.emplace(reference, (*mapping)->view()).second)
      return invalid("SpatialMapping import table contains a duplicate");
    if (auto target = mappingTargetClass(fabric, (*mapping)->view()); !target)
      return target.takeError();
  }

  std::vector<SystemThreadExecutionBindingView> threadBindings;
  std::vector<SystemGraphExecutionBindingView> graphBindings;
  std::map<std::string, std::size_t> threadByKey;
  std::set<std::string> seenKeys;
  for (mlir::Operation &operation : parsed->root.getBody().front()) {
    if (executionControl.stopRequested())
      return interrupted();
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
      SystemThreadExecutionBindingView view{
          *key,
          {},
          std::nullopt,
          ::mapping::SystemBindingRelationKind::PresburgerPartition,
          {}};
      view.relationKind = binding.getRelationKind();
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
      for (auto entry : binding.getBody()
                            .front()
                            .getOps<::mapping::ThreadStableKeyEntryOp>()) {
        auto stableKey = ::dataflow::decodeDynamicWorkStableItemKey(
            unsignedBytes(entry.getStableKey()));
        if (!stableKey)
          return stableKey.takeError();
        auto target = decodeFabric<::loom::fabric::AccCoreOccurrenceRef>(
            entry.getTarget());
        if (!target)
          return target.takeError();
        if (auto targetClass = coreTargetClass(fabric, *target); !targetClass)
          return targetClass.takeError();
        view.stableKeyEntries.push_back({*stableKey, *target});
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
      if (view.relationKind != expectedIt->second->relationKind)
        return invalid("ThreadExecutionBinding relation kind differs from D");
      if (view.relationKind ==
          ::mapping::SystemBindingRelationKind::StableKeyLookup) {
        if (!view.clauses.empty() || view.defaultTarget)
          return invalid("DynamicWork binding contains a Presburger branch");
        if (llvm::Error error = validateStableRelation(
                view.stableKeyEntries, expectedIt->second->stableItemKeys))
          return std::move(error);
      } else {
        if (!view.stableKeyEntries.empty())
          return invalid("dense binding contains a stable-key entry");
        if (llvm::Error error =
                validateRelation(cells, view.defaultTarget.has_value(),
                                 expectedIt->second->legalDomain))
          return std::move(error);
      }
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
    SystemGraphExecutionBindingView view{
        *key,
        {},
        std::nullopt,
        ::mapping::SystemBindingRelationKind::PresburgerPartition,
        {}};
    view.relationKind = binding.getRelationKind();
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
    for (auto entry :
         binding.getBody().front().getOps<::mapping::GraphStableKeyEntryOp>()) {
      auto stableKey = ::dataflow::decodeDynamicWorkStableItemKey(
          unsignedBytes(entry.getStableKey()));
      if (!stableKey)
        return stableKey.takeError();
      const std::uint64_t ordinal = entry.getTarget().getOrdinal();
      if (ordinal >= imports.size())
        return invalid(
            "graph stable key names an absent SpatialMapping import");
      view.stableKeyEntries.push_back({*stableKey, imports[ordinal]});
    }
    if (binding.getDefaultTarget()) {
      const std::uint64_t ordinal = binding.getDefaultTarget()->getOrdinal();
      if (ordinal >= imports.size())
        return invalid("graph default names an absent SpatialMapping import");
      view.defaultTarget = imports[ordinal];
    }
    if (view.relationKind != expectedIt->second->relationKind)
      return invalid("GraphExecutionBinding relation kind differs from D");
    if (view.relationKind ==
        ::mapping::SystemBindingRelationKind::StableKeyLookup) {
      if (!view.clauses.empty() || view.defaultTarget)
        return invalid("DynamicWork graph binding has a Presburger branch");
      if (llvm::Error error = validateStableRelation(
              view.stableKeyEntries, expectedIt->second->stableItemKeys))
        return std::move(error);
    } else {
      if (!view.stableKeyEntries.empty())
        return invalid("dense graph binding contains a stable-key entry");
      if (llvm::Error error =
              validateRelation(cells, view.defaultTarget.has_value(),
                               expectedIt->second->legalDomain))
        return std::move(error);
    }
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
    for (const auto &entry : binding.stableKeyEntries)
      selectedImports.push_back(entry.target);
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
    const ArtifactStore &store,
    const SpatialMappingImportContext *spatialMappings,
    std::shared_ptr<const SystemMappingClosureProjection> *verifiedClosure,
    ExecutionControlView executionControl) {
  if (executionControl.stopRequested())
    return cancelled();
  auto canonical = writeCanonicalSystemMappingAssembly(root);
  if (!canonical)
    return canonical.takeError();
  if (finalizeArtifactIdentity(mappingArtifactSchema, *canonical) !=
      mappingIdentity)
    return invalid("mapping identity does not match canonical bytes");
  auto imports = decodeSpatialMappingImports(root);
  if (!imports)
    return imports.takeError();
  std::optional<SpatialMappingImportContext> ownedSpatialMappings;
  if (!spatialMappings) {
    auto built = buildSpatialMappingImportContext(*imports, store);
    if (!built)
      return built.takeError();
    ownedSpatialMappings.emplace(std::move(*built));
    spatialMappings = &*ownedSpatialMappings;
  }
  if (llvm::Error error =
          validateSpatialMappingImportContext(*imports, *spatialMappings))
    return std::move(error);

  auto execution = strictImportSystemExecutionBindings(
      *canonical, dataflow, fabric, store, spatialMappings, executionControl);
  if (!execution) {
    if (executionControl.stopRequested()) {
      llvm::consumeError(execution.takeError());
      return cancelled();
    }
    return execution.takeError();
  }
  if (executionControl.stopRequested())
    return cancelled();
  auto closure = detail::importSystemMappingClosure(
      root, dataflow, fabric, *execution, *spatialMappings, executionControl);
  if (!closure) {
    if (executionControl.stopRequested()) {
      llvm::consumeError(closure.takeError());
      return cancelled();
    }
    return closure.takeError();
  }
  if (executionControl.stopRequested())
    return cancelled();
  auto physicalDemand = detail::verifySystemMappingCapacity(
      dataflow, fabric, *execution, closure->services, closure->resourceUses,
      closure->resourceUseActivationKeys, *spatialMappings, executionControl);
  if (!physicalDemand) {
    if (executionControl.stopRequested()) {
      llvm::consumeError(physicalDemand.takeError());
      return cancelled();
    }
    return physicalDemand.takeError();
  }
  if (executionControl.stopRequested())
    return cancelled();
  if (llvm::Error error = detail::verifySystemMappingHandshakeClosure(
          dataflow, fabric, *execution, closure->services, *spatialMappings,
          executionControl)) {
    if (executionControl.stopRequested()) {
      llvm::consumeError(std::move(error));
      return cancelled();
    }
    return std::move(error);
  }
  if (executionControl.stopRequested())
    return cancelled();

  (void)physicalDemand;
  SystemMappingView result(mappingIdentity, dataflow.identity(),
                           fabric.artifact().identity(), std::move(*execution),
                           std::move(closure->services),
                           std::move(closure->resourceUses));
  auto projected = projectSystemMappingClosure(
      dataflow, fabric, result, store, spatialMappings, executionControl);
  if (!projected) {
    if (executionControl.stopRequested()) {
      llvm::consumeError(projected.takeError());
      return cancelled();
    }
    return projected.takeError();
  }
  if (executionControl.stopRequested())
    return cancelled();
  auto progress =
      deriveSystemMappingProgressClosure(dataflow, fabric, *projected);
  if (!progress) {
    if (executionControl.stopRequested()) {
      llvm::consumeError(progress.takeError());
      return cancelled();
    }
    return progress.takeError();
  }
  if (executionControl.stopRequested())
    return cancelled();
  switch (progress->kind) {
  case MappingProgressClosureKind::ProvenNoClosedWaitSet:
    break;
  case MappingProgressClosureKind::ProvenClosedWaitSet:
    return llvm::make_error<SystemMappingRejectedError>(
        SystemMappingClosureFindingKind::HardProgressViolation,
        "selected System physical demand contains a closed wait set");
  case MappingProgressClosureKind::ProofNotEstablished:
    return llvm::make_error<SystemMappingIncompleteError>(
        SystemMappingIncompleteReason::ProofNotEstablished,
        "system route, service, and resource progress proof is not "
        "established");
  }
  if (verifiedClosure)
    *verifiedClosure = std::make_shared<const SystemMappingClosureProjection>(
        std::move(*projected));
  return result;
}

SystemMappingBaseVerification verifySystemMappingBase(
    ::mapping::SystemOp source,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ArtifactStore &store) {
  auto assembly = detail::prepareCanonicalSystemMappingAssembly(source);
  if (!assembly) {
    std::string diagnostic = llvm::toString(assembly.takeError());
    return RejectedSystemMappingBase{
        SystemMappingClosureFindingKind::InvalidClosure, std::move(diagnostic)};
  }
  const ArtifactIdentity identity =
      finalizeArtifactIdentity(mappingArtifactSchema, assembly->bytes);
  auto view = importSystemMappingView(
      identity, mlir::cast<::mapping::SystemOp>(assembly->root.get()), dataflow,
      fabric, store, nullptr, nullptr, {});
  if (!view) {
    std::optional<SystemMappingBaseVerification> typed;
    llvm::Error remaining = llvm::handleErrors(
        view.takeError(),
        [&](const SystemMappingIncompleteError &error) {
          typed = IncompleteSystemMappingBase{error.reason(),
                                              error.diagnostic().str()};
        },
        [&](const SystemMappingRejectedError &error) {
          typed = RejectedSystemMappingBase{error.finding(),
                                            error.diagnostic().str()};
        });
    if (typed) {
      if (remaining)
        return InternalSystemMappingBaseError{
            "typed System Mapping result was mixed with an unclassified "
            "error: " +
            llvm::toString(std::move(remaining))};
      return std::move(*typed);
    }
    return RejectedSystemMappingBase{
        SystemMappingClosureFindingKind::InvalidClosure,
        llvm::toString(std::move(remaining))};
  }
  return VerifiedSystemMappingBase{};
}

llvm::Expected<FinalizedSystemMapping>
finalizeSystemMapping(::mapping::SystemOp source,
                      const ::dataflow::CanonicalDataflowProgramView &dataflow,
                      const ::loom::fabric::FabricSystemRootView &fabric,
                      const SystemMappingConstraintSetView &constraints,
                      const ArtifactStore &store,
                      const SpatialMappingImportContext *spatialMappings,
                      ExecutionControlView executionControl) {
  if (executionControl.stopRequested())
    return cancelled();
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
    if (executionControl.stopRequested())
      return cancelled();
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
  std::shared_ptr<const SystemMappingClosureProjection> verifiedClosure;
  auto view = importSystemMappingView(identity, root, dataflow, fabric, store,
                                      spatialMappings, &verifiedClosure,
                                      executionControl);
  if (!view)
    return view.takeError();
  if (!verifiedClosure)
    return invalid("strict import did not publish its verified closure");
  if (constraints.rootThreadLaunches() !=
      view->executionBindings().rootThreadLaunches())
    return invalid("System constraint root scope does not match Mapping");
  if (llvm::Error error =
          admitSystemMappingConstraints(dataflow, fabric, constraints, *view))
    return std::move(error);

  if (executionControl.stopRequested())
    return cancelled();

  auto stored = store.put(mappingArtifactSchema, assembly->bytes);
  if (!stored)
    return stored.takeError();
  if (*stored != identity)
    return invalid("ArtifactStore returned a different Mapping identity");
  ArtifactRootReference reference{mappingArtifactSchema.identity.str(),
                                  mappingArtifactSchema.version, identity};
  return FinalizedSystemMapping(std::move(reference),
                                std::move(assembly->bytes), std::move(*view),
                                std::move(verifiedClosure));
}

llvm::Expected<FinalizedSystemMapping>
importSystemMapping(const ArtifactRootReference &reference,
                    const ArtifactStore &store) {
  if (reference.schemaIdentity != mappingArtifactSchema.identity ||
      reference.schemaVersion != mappingArtifactSchema.version)
    return invalid("root reference has the wrong Mapping schema");
  if (currentSystemMappingImportSession &&
      !currentSystemMappingImportSession->owns(store))
    return invalid("SystemMapping import session crosses its ArtifactStore "
                   "verification domain");
  if (currentSystemMappingImportSession)
    if (auto cached = currentSystemMappingImportSession->find(reference))
      return *cached;

  const auto begin = MonotonicClock::now();
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
  std::shared_ptr<const SystemMappingClosureProjection> verifiedClosure;
  auto view =
      importSystemMappingView(reference.artifact, parsed->root, *dataflowView,
                              *system, store, nullptr, &verifiedClosure, {});
  if (!view)
    return view.takeError();
  if (!verifiedClosure)
    return invalid("strict import did not publish its verified closure");
  FinalizedSystemMapping value(reference, std::move(*canonical),
                               std::move(*view), std::move(verifiedClosure));
  auto imported =
      std::make_shared<const FinalizedSystemMapping>(std::move(value));
  if (!currentSystemMappingImportSession)
    return *imported;
  const std::uint64_t constructionNanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          MonotonicClock::now() - begin)
          .count();
  auto cached = currentSystemMappingImportSession->insert(
      reference, imported, retainedSystemMappingBytes(*imported),
      constructionNanoseconds, deterministicSystemMappingWork(*imported));
  return *cached;
}

namespace {

template <typename Ref, typename Attr>
llvm::Expected<Attr> remapFabricAttribute(
    Attr attribute,
    const ::loom::fabric::FabricSystemReferenceRemapper &remapper) {
  auto decoded = decodeFabric<Ref>(attribute);
  if (!decoded)
    return decoded.takeError();
  auto mapped = remapper.remap(*decoded);
  if (!mapped)
    return mapped.takeError();
  return Attr::get(attribute.getContext(),
                   denseBytes(attribute.getContext(),
                              ::loom::fabric::canonicalFabricBytes(*mapped)));
}

llvm::Expected<mlir::ArrayAttr> remapTransformPath(
    mlir::ArrayAttr path,
    const ::loom::fabric::FabricSystemReferenceRemapper &remapper) {
  llvm::SmallVector<mlir::Attribute> result;
  result.reserve(path.size());
  for (mlir::Attribute raw : path) {
    auto transform =
        mlir::dyn_cast<::mapping::SystemServiceTransformRefAttr>(raw);
    if (!transform)
      return invalid("System transform path has a non-transform reference");
    auto mapped =
        remapFabricAttribute<::loom::fabric::SystemServiceTransformRef>(
            transform, remapper);
    if (!mapped)
      return mapped.takeError();
    result.push_back(*mapped);
  }
  return mlir::ArrayAttr::get(path.getContext(), result);
}

llvm::Expected<mlir::Attribute> remapServicePlanElement(
    mlir::Attribute element,
    const ::loom::fabric::FabricSystemReferenceRemapper &remapper) {
  if (auto memory =
          mlir::dyn_cast<::mapping::MemoryRegionElementKeyAttr>(element)) {
    auto region =
        remapFabricAttribute<::loom::fabric::FabricMemoryServiceRegionRef>(
            memory.getServiceRegion(), remapper);
    if (!region)
      return region.takeError();
    auto path = remapTransformPath(memory.getTransformPath(), remapper);
    if (!path)
      return path.takeError();
    return ::mapping::MemoryRegionElementKeyAttr::get(
        element.getContext(), memory.getLogicalMemory(), memory.getInterval(),
        *region, *path);
  }
  if (auto consistency =
          mlir::dyn_cast<::mapping::ConsistencyElementKeyAttr>(element)) {
    auto domain =
        remapFabricAttribute<::loom::fabric::MemoryConsistencyDomainRef>(
            consistency.getConsistencyDomain(), remapper);
    if (!domain)
      return domain.takeError();
    return ::mapping::ConsistencyElementKeyAttr::get(
        element.getContext(), consistency.getFence(), *domain);
  }
  if (mlir::isa<::mapping::TransferLegElementKeyAttr>(element))
    return element;
  return invalid("System ResourceUse has an unknown service-plan element");
}

llvm::Expected<mlir::Attribute> remapSystemResourceOwner(
    mlir::Attribute owner,
    const ::loom::fabric::FabricSystemReferenceRemapper &remapper) {
  if (auto instruction =
          mlir::dyn_cast<::mapping::InstructionExecutionResourceOwnerRefAttr>(
              owner)) {
    auto context =
        remapFabricAttribute<::loom::fabric::InstructionCoreContextRef>(
            instruction.getInstructionContext(), remapper);
    if (!context)
      return context.takeError();
    return ::mapping::InstructionExecutionResourceOwnerRefAttr::get(
        owner.getContext(), instruction.getRoot(), *context);
  }
  if (auto service =
          mlir::dyn_cast<::mapping::ServicePlanElementRefAttr>(owner)) {
    auto element = remapServicePlanElement(service.getElement(), remapper);
    if (!element)
      return element.takeError();
    return ::mapping::ServicePlanElementRefAttr::get(
        owner.getContext(), service.getService(), service.getPlanOrdinal(),
        *element);
  }
  return invalid("System ResourceUse has an unknown owner reference");
}

llvm::Expected<::mapping::ServicePlanSelectionKeyAttr>
remapServicePlanSelectionKey(
    ::mapping::ServicePlanSelectionKeyAttr attribute,
    const ArtifactIdentity &dataflowIdentity,
    const ::loom::fabric::FabricSystemReferenceRemapper &remapper) {
  auto decoded = decodeServicePlanSelectionKey(
      unsignedBytes(attribute.getRecord()), dataflowIdentity);
  if (!decoded)
    return decoded.takeError();
  llvm::Error remapError = llvm::Error::success();
  std::visit(
      [&](auto &context) {
        auto mapped = remapper.remap(context.accCore);
        if (!mapped)
          remapError = mapped.takeError();
        else
          context.accCore = *mapped;
      },
      decoded->context);
  if (remapError)
    return std::move(remapError);
  auto encoded = encodeServicePlanSelectionKey(dataflowIdentity, *decoded);
  if (!encoded)
    return encoded.takeError();
  return ::mapping::ServicePlanSelectionKeyAttr::get(
      attribute.getContext(), denseBytes(attribute.getContext(), *encoded));
}

llvm::Error remapSystemMappingFabricReferences(
    ::mapping::SystemOp root, const ArtifactIdentity &dataflowIdentity,
    const ::loom::fabric::FabricSystemReferenceRemapper &remapper) {
  llvm::Error error = llvm::Error::success();
  root.walk([&](mlir::Operation *operation) {
    if (error)
      return mlir::WalkResult::interrupt();
    if (auto binding =
            mlir::dyn_cast<::mapping::ThreadExecutionBindingOp>(operation)) {
      if (auto target = binding.getDefaultTarget()) {
        auto mapped =
            remapFabricAttribute<::loom::fabric::AccCoreOccurrenceRef>(
                *target, remapper);
        if (!mapped)
          error = mapped.takeError();
        else
          binding->setAttr("default_target", *mapped);
      }
    } else if (auto clause =
                   mlir::dyn_cast<::mapping::ThreadPresburgerClauseOp>(
                       operation)) {
      auto mapped = remapFabricAttribute<::loom::fabric::AccCoreOccurrenceRef>(
          clause.getTarget(), remapper);
      if (!mapped)
        error = mapped.takeError();
      else
        clause->setAttr("target", *mapped);
    } else if (auto target =
                   mlir::dyn_cast<::mapping::MemoryRegionTargetOp>(operation)) {
      auto region =
          remapFabricAttribute<::loom::fabric::FabricMemoryServiceRegionRef>(
              target.getServiceRegion(), remapper);
      if (!region) {
        error = region.takeError();
      } else {
        auto path = remapTransformPath(target.getTransformPath(), remapper);
        if (!path)
          error = path.takeError();
        else {
          target->setAttr("service_region", *region);
          target->setAttr("transform_path", *path);
        }
      }
    } else if (auto exposure =
                   mlir::dyn_cast<::mapping::SystemMemoryExposureOp>(
                       operation)) {
      auto mapped =
          remapFabricAttribute<::loom::fabric::SubordinateEndpointRef>(
              exposure.getTerminal(), remapper);
      if (!mapped)
        error = mapped.takeError();
      else
        exposure->setAttr("terminal", *mapped);
    } else if (auto consistency =
                   mlir::dyn_cast<::mapping::ConsistencyTargetOp>(operation)) {
      auto mapped =
          remapFabricAttribute<::loom::fabric::MemoryConsistencyDomainRef>(
              consistency.getConsistencyDomain(), remapper);
      if (!mapped)
        error = mapped.takeError();
      else
        consistency->setAttr("consistency_domain", *mapped);
    } else if (auto selection =
                   mlir::dyn_cast<::mapping::ServicePlanSelectionOp>(
                       operation)) {
      auto key = remapServicePlanSelectionKey(selection.getKey(),
                                              dataflowIdentity, remapper);
      if (!key)
        error = key.takeError();
      else
        selection->setAttr("key", *key);
    } else if (auto leg = mlir::dyn_cast<::mapping::TransferLegRealizationOp>(
                   operation)) {
      auto mapped =
          remapFabricAttribute<::loom::fabric::FabricTransportEndpointRef>(
              leg.getRootEndpoint(), remapper);
      if (!mapped)
        error = mapped.takeError();
      else
        leg->setAttr("root_endpoint", *mapped);
    } else if (auto node =
                   mlir::dyn_cast<::mapping::SystemRouteNodeOp>(operation)) {
      auto mapped =
          remapFabricAttribute<::loom::fabric::FabricPhysicalTraversalRef>(
              node.getIncomingTraversal(), remapper);
      if (!mapped)
        error = mapped.takeError();
      else
        node->setAttr("incoming_traversal", *mapped);
    } else if (auto use = mlir::dyn_cast<::mapping::ResourceUseOp>(operation)) {
      if (!mlir::isa<::mapping::SystemOp>(use->getParentOp()))
        return mlir::WalkResult::advance();
      auto owner = remapSystemResourceOwner(use.getOwner(), remapper);
      if (!owner) {
        error = owner.takeError();
      } else {
        auto site = remapFabricAttribute<::loom::fabric::FabricUsePatternRef>(
            use.getUseSite(), remapper);
        if (!site)
          error = site.takeError();
        else {
          use->setAttr("owner", *owner);
          use->setAttr("use_site", *site);
        }
      }
    }
    return error ? mlir::WalkResult::interrupt() : mlir::WalkResult::advance();
  });
  return error;
}

} // namespace

llvm::Expected<FinalizedSystemMapping> rebaseSystemMapping(
    const FinalizedSystemMapping &parent,
    const ::loom::fabric::FabricSystemRootView &childFabric,
    llvm::ArrayRef<ArtifactRootReference> childSpatialMappings,
    llvm::ArrayRef<::loom::fabric::FabricSystemEntityCorrespondence> entities,
    llvm::ArrayRef<::loom::fabric::FabricSystemTransferPatternCorrespondence>
        transferPatterns,
    const SystemMappingConstraintSetView &childConstraints,
    const ArtifactStore &store,
    const SpatialMappingImportContext *spatialMappings) {
  if (childSpatialMappings.size() !=
      parent.view().executionBindings().spatialMappingImports().size())
    return invalid("System Mapping rebase changed the Spatial import count");
  auto parsed = parseSystemRoot(parent.canonicalBytes());
  if (!parsed)
    return parsed.takeError();
  auto remapper = ::loom::fabric::FabricSystemReferenceRemapper::get(
      entities, transferPatterns);
  if (!remapper)
    return remapper.takeError();
  auto dataflowIdentity = ArtifactIdentity::fromBytes(
      unsignedBytes(parsed->root.getDataflow().getRecord()));
  if (!dataflowIdentity)
    return dataflowIdentity.takeError();
  if (llvm::Error error = remapSystemMappingFabricReferences(
          parsed->root, *dataflowIdentity, *remapper))
    return std::move(error);
  parsed->root.setFabricAttr(::mapping::ArtifactIdentityAttr::get(
      parsed->context.get(),
      denseBytes(parsed->context.get(),
                 childFabric.artifact().identity().bytes())));
  llvm::SmallVector<mlir::Attribute> imports;
  imports.reserve(childSpatialMappings.size());
  for (const ArtifactRootReference &reference : childSpatialMappings)
    imports.push_back(::mapping::ArtifactRootReferenceAttr::get(
        parsed->context.get(),
        denseBytes(parsed->context.get(),
                   encodeArtifactRootReference(reference))));
  parsed->root.setSpatialMappingImportsAttr(
      mlir::ArrayAttr::get(parsed->context.get(), imports));
  auto dataflow = ::dataflow::importCanonicalDataflow(
      {::dataflow::canonicalDataflowSchema.identity.str(),
       ::dataflow::canonicalDataflowSchema.version,
       parent.view().dataflowIdentity()},
      store);
  if (!dataflow)
    return dataflow.takeError();
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return dataflowView.takeError();
  return finalizeSystemMapping(parsed->root, *dataflowView, childFabric,
                               childConstraints, store, spatialMappings);
}

} // namespace loom::mapping
