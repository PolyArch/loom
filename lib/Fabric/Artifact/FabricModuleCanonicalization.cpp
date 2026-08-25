#include "FabricModuleCanonicalization.h"

#include "Fabric/IR/Elaboration.h"
#include "Fabric/IR/FabricCanonicalEntity.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/ModuleDomain.h"
#include "FabricCanonicalLabeling.h"
#include "FabricFuCapabilityDerivation.h"
#include "FabricModuleCanonicalPayload.h"
#include "FabricModuleDomainMaterialization.h"
#include "FabricModuleDomainNormalization.h"
#include "FabricResourceContractFinalization.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;

namespace loom::fabric {
namespace {

constexpr llvm::StringLiteral canonicalRootName("__loom_fabric_root");

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

llvm::Error
reorderCanonicalGraphRegions(::fabric::ModuleOp root,
                             const detail::FabricCanonicalLabeling &labeling) {
  llvm::DenseMap<Operation *, std::uint64_t> rank;
  for (auto [index, operation] :
       llvm::enumerate(labeling.canonicalOperationOrder))
    rank[operation] = index;

  llvm::SmallVector<Block *> blocks;
  root->walk([&](Operation *operation) {
    if (!isa<::fabric::ModuleOp, ::fabric::PeOp, ::fabric::FuOp>(operation))
      return;
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        blocks.push_back(&block);
  });

  for (Block *block : blocks) {
    llvm::SmallVector<Operation *> ordered;
    Operation *terminator = nullptr;
    const bool fuDefinition = isa<::fabric::FuOp>(block->getParentOp());
    for (Operation &operation : *block) {
      if (operation.hasTrait<OpTrait::IsTerminator>()) {
        terminator = &operation;
        continue;
      }
      const bool known =
          fuDefinition
              ? labeling.definitionFuNodeOrdinalByOperation.count(&operation)
              : rank.count(&operation);
      if (!known)
        return invalid("canonical operation order omits a graph operation");
      ordered.push_back(&operation);
    }
    llvm::sort(ordered, [&](Operation *left, Operation *right) {
      if (fuDefinition)
        return labeling.definitionFuNodeOrdinalByOperation.lookup(left) <
               labeling.definitionFuNodeOrdinalByOperation.lookup(right);
      return rank.lookup(left) < rank.lookup(right);
    });
    for (Operation *operation : ordered) {
      if (terminator)
        operation->moveBefore(terminator);
      else
        operation->moveBefore(block, block->end());
    }
  }
  return llvm::Error::success();
}

std::optional<FabricEntityKind> moduleOccurrenceKind(Operation *operation) {
  if (isa<::fabric::ModuleOp>(operation))
    return FabricEntityKind::FabricModuleTemplate;
  if (isa<::fabric::PeOp>(operation))
    return FabricEntityKind::FabricPeOccurrence;
  if (isa<::fabric::FuOp>(operation))
    return FabricEntityKind::FabricFuOccurrence;
  if (isa<::fabric::MemOp>(operation))
    return FabricEntityKind::FabricMemoryOccurrence;
  if (isa<::fabric::SwitchOp>(operation))
    return FabricEntityKind::FabricSwitchOccurrence;
  if (isa<::fabric::FifoOp>(operation))
    return FabricEntityKind::FabricFifoOccurrence;
  if (isa<::fabric::BoundaryOp>(operation))
    return FabricEntityKind::FabricBoundaryOccurrence;
  return std::nullopt;
}

llvm::Expected<std::vector<FabricModuleEntityCorrespondence>>
projectModuleEntityCorrespondence(
    const std::map<Operation *, FabricModuleEntityReference> &authored,
    const detail::FabricCanonicalLabeling &labeling) {
  std::vector<FabricModuleEntityCorrespondence> result;
  std::set<std::pair<FabricEntityKind, FabricEntityId>> sourceKeys;
  std::set<std::pair<FabricEntityKind, FabricEntityId>> targetKeys;
  std::map<Operation *, FabricEntityId> targetByOperation;
  std::map<FabricEntityKind, std::uint64_t> targetOrdinals;
  for (const detail::FabricEntityCarrier &carrier : labeling.carriers) {
    if (!carrier.op)
      continue;
    const auto priorTarget = targetByOperation.find(carrier.op);
    if (priorTarget != targetByOperation.end()) {
      if (priorTarget->second != carrier.id)
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "fabric_module_correspondence_invalid: one operation has "
            "multiple canonical entity IDs");
      continue;
    }
    targetByOperation.emplace(carrier.op, carrier.id);
    const auto source = authored.find(carrier.op);
    if (source == authored.end() || source->second.kind != carrier.kind)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "fabric_module_correspondence_invalid: Module canonical entity has "
          "no authored correspondence kind=" +
              llvm::Twine(static_cast<std::uint32_t>(carrier.kind)) +
              " authored_count=" + llvm::Twine(authored.size()));
    const auto sourceKey =
        std::make_pair(source->second.kind, source->second.occurrenceOrdinal);
    const auto targetOrdinal = targetOrdinals[carrier.kind]++;
    const auto targetKey = std::make_pair(carrier.kind, targetOrdinal);
    const bool sourceInserted = sourceKeys.insert(sourceKey).second;
    const bool targetInserted = targetKeys.insert(targetKey).second;
    if (!sourceInserted || !targetInserted)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "fabric_module_correspondence_invalid: Module entity "
          "correspondence is not one-to-one");
    result.push_back(
        {source->second, {carrier.kind, carrier.id, targetOrdinal}});
  }
  if (result.size() != authored.size())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "fabric_module_correspondence_invalid: Module entity "
        "correspondence is incomplete");
  llvm::sort(result, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.source.kind, lhs.source.occurrenceOrdinal) <
           std::tie(rhs.source.kind, rhs.source.occurrenceOrdinal);
  });
  return result;
}

} // namespace

llvm::Expected<detail::CanonicalFabricModuleCandidate>
detail::buildCanonicalFabricModuleCandidate(
    ::fabric::ModuleOp source,
    const ::fabric::ModuleDomainAuthoringRelation *domainRelation,
    bool captureCorrespondence) {
  auto sourceModule = source->getParentOfType<ModuleOp>();
  if (!sourceModule || source->getParentOp() != sourceModule.getOperation())
    return invalid("the selected Fabric Module must be top-level");
  if (failed(verify(sourceModule)))
    return invalid("the authoring module does not verify");

  IRMapping cloneMapping;
  OwningOpRef<ModuleOp> scratch(
      cast<ModuleOp>(sourceModule->clone(cloneMapping)));
  Operation *clonedOperation =
      SymbolTable::lookupSymbolIn(*scratch, source.getSymNameAttr());
  auto clonedRoot = dyn_cast_or_null<::fabric::ModuleOp>(clonedOperation);
  if (!clonedRoot)
    return invalid("the selected Fabric root was not cloned");

  std::map<Operation *, FabricModuleEntityReference> authoredEntities;
  if (captureCorrespondence) {
    std::map<FabricEntityKind, std::uint64_t> authoredOrdinals;
    clonedRoot->walk([&](Operation *operation) {
      const auto kind = moduleOccurrenceKind(operation);
      if (!kind)
        return;
      auto id = operation->getAttrOfType<::fabric::EntityIdAttr>(
          ::fabric::kEntityIdAttrName);
      if (id)
        authoredEntities.emplace(
            operation, FabricModuleEntityReference{*kind, id.getId(),
                                                   authoredOrdinals[*kind]++});
    });
    if (authoredEntities.empty())
      return invalid(
          "Module correspondence capture requires canonical parent IDs");
  }

  std::optional<::fabric::ModuleDomainAuthoringRelation> remappedDomain;
  if (domainRelation && domainRelation->hasDomainAuthoring()) {
    auto remapped = domainRelation->remap(cloneMapping);
    if (!remapped)
      return remapped.takeError();
    remappedDomain = std::move(*remapped);
  }

  if (llvm::Error error = detail::stripFabricModuleAuthoringState(clonedRoot))
    return std::move(error);
  bool hasInstances = false;
  clonedRoot->walk([&](::fabric::InstantiateOp) { hasInstances = true; });
  if (hasInstances) {
    LogicalResult elaborated =
        remappedDomain
            ? ::fabric::elaborateInstances(clonedRoot, *remappedDomain)
            : ::fabric::elaborateInstances(clonedRoot);
    if (failed(elaborated))
      return invalid("fabric.instantiate elaboration failed");
  }
  if (llvm::Error error =
          detail::eraseElaboratedFabricModuleDeclarations(clonedRoot))
    return std::move(error);

  for (Operation &operation :
       llvm::make_early_inc_range(scratch->getBody()->getOperations()))
    if (&operation != clonedRoot.getOperation())
      operation.erase();
  clonedRoot.setSymName(canonicalRootName);
  if (failed(verify(*scratch)))
    return invalid("the root-complete Fabric candidate does not verify");

  auto normalizedDomain =
      remappedDomain
          ? detail::normalizeFabricModuleDomain(clonedRoot, *remappedDomain)
          : detail::buildDefaultFabricModuleDomain(clonedRoot);
  if (!normalizedDomain)
    return normalizedDomain.takeError();

  auto preliminary = detail::computeFabricModuleCanonicalLabeling(
      clonedRoot, *normalizedDomain);
  if (!preliminary)
    return preliminary.takeError();
  if (llvm::Error error =
          detail::materializeFabricResourceContracts(clonedRoot, *preliminary))
    return std::move(error);
  if (failed(verify(*scratch)))
    return invalid("the complete Fabric resource contracts do not verify");

  auto labeling = detail::computeFabricModuleCanonicalLabeling(
      clonedRoot, *normalizedDomain);
  if (!labeling)
    return labeling.takeError();
  if (llvm::Error error = reorderCanonicalGraphRegions(clonedRoot, *labeling))
    return std::move(error);
  if (llvm::Error error =
          detail::materializeFabricCanonicalFuCapabilityDomains(*labeling))
    return std::move(error);
  auto reordered = detail::computeCanonicalFabricModulePayloadLabeling(
      clonedRoot, *normalizedDomain);
  if (!reordered)
    return reordered.takeError();
  std::vector<FabricModuleEntityCorrespondence> entities;
  if (captureCorrespondence) {
    auto projected =
        projectModuleEntityCorrespondence(authoredEntities, *reordered);
    if (!projected)
      return projected.takeError();
    entities = std::move(*projected);
  }
  if (llvm::Error error = detail::materializeFabricCanonicalIds(*reordered))
    return std::move(error);
  if (llvm::Error error = detail::materializeFabricModuleDomainRelation(
          clonedRoot, *normalizedDomain, *reordered))
    return std::move(error);
  if (failed(verify(*scratch)))
    return invalid("canonical Fabric IDs produced invalid IR");
  return detail::CanonicalFabricModuleCandidate{std::move(scratch),
                                                std::move(entities)};
}

} // namespace loom::fabric
