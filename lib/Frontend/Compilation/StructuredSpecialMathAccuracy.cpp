#include "Frontend/Compilation/StructuredSpecialMathAccuracy.h"

#include "Dataflow/IR/OperationSchema.h"
#include "Frontend/IR/LoomOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom::frontend {
namespace {

constexpr llvm::StringLiteral decisionSchema =
    "loom.structured_special_math_accuracy.decision.1.0";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "structured_special_math_accuracy_invalid: " +
                                     message);
}

bool isSelectedSpatialOperation(mlir::Operation *operation) {
  for (mlir::Operation *owner = operation->getParentOp(); owner;
       owner = owner->getParentOp()) {
    if (llvm::isa<loom::SpatialRegionOp>(owner))
      return true;
    if (llvm::isa<mlir::FunctionOpInterface>(owner))
      return false;
  }
  return false;
}

llvm::Expected<bool> permitsApproximation(mlir::Operation *operation) {
  auto fastMath =
      llvm::dyn_cast<mlir::arith::ArithFastMathInterface>(operation);
  if (!fastMath)
    return invalid("special-math operation has no fast-math interface");
  mlir::arith::FastMathFlags flags = mlir::arith::FastMathFlags::none;
  if (mlir::arith::FastMathFlagsAttr attr = fastMath.getFastMathFlagsAttr())
    flags = attr.getValue();
  using Bits = std::underlying_type_t<mlir::arith::FastMathFlags>;
  return (static_cast<Bits>(flags) &
          static_cast<Bits>(mlir::arith::FastMathFlags::afn)) != 0;
}

llvm::Expected<std::optional<SpecialMathAccuracyTier>>
selectedAccuracy(mlir::Operation *operation, bool approximationPermitted) {
  mlir::Attribute attribute =
      operation->getDiscardableAttr(kSpecialMathAccuracyAttrName);
  if (!attribute)
    return std::optional<SpecialMathAccuracyTier>{};
  auto spelling = llvm::dyn_cast<mlir::StringAttr>(attribute);
  if (!spelling)
    return invalid("special-math accuracy attribute is not a string");
  std::optional<SpecialMathAccuracyTier> tier =
      symbolizeSpecialMathAccuracyTier(spelling.getValue());
  if (!tier)
    return invalid("special-math accuracy attribute has an unknown tier");
  if (*tier != SpecialMathAccuracyTier::CorrectlyRounded &&
      !approximationPermitted)
    return invalid("relaxed special-math accuracy is selected without afn");
  return tier;
}

struct MaterializedAccuracyProjection final {
  StructuredProgramCandidate structuredProgram;
  std::vector<StructuredEntityRef> trackedBlocks;
  std::optional<StructuredEntityRef> trackedSpatialRegion;
  std::vector<StructuredOperationSourceProvenance> sourceProvenance;
};

llvm::Expected<MaterializedAccuracyProjection> materializeDecision(
    const StructuredProgramCandidate &parent,
    const StructuredSpecialMathAccuracyDecision &decision,
    llvm::ArrayRef<StructuredEntityRef> trackedBlockReferences,
    std::optional<StructuredEntityRef> trackedSpatialRegion,
    llvm::ArrayRef<StructuredOperationSourceProvenance> sourceProvenance) {
  auto domain = enumerateStructuredSpecialMathAccuracyDecisions(parent);
  if (!domain)
    return domain.takeError();
  if (!llvm::is_contained(*domain, decision))
    return invalid("decision is outside the exact parent decision domain");

  auto parentView = parent.view();
  if (!parentView)
    return parentView.takeError();
  auto selected = parentView->resolve(decision.operation);
  if (!selected)
    return selected.takeError();

  mlir::IRMapping mapping;
  mlir::OwningOpRef<mlir::ModuleOp> clone(
      llvm::cast<mlir::ModuleOp>(parent.module()->clone(mapping)));
  mlir::Operation *trackedSpatialOperation = nullptr;
  if (trackedSpatialRegion) {
    if (trackedSpatialRegion->parent != parent.identity() ||
        trackedSpatialRegion->kind != StructuredEntityKind::Operation)
      return invalid("tracked Spatial region has the wrong Structured owner");
    auto entity = parentView->resolve(*trackedSpatialRegion);
    if (!entity)
      return entity.takeError();
    if (!llvm::isa_and_nonnull<loom::SpatialRegionOp>(entity->operation))
      return invalid("tracked operation is not a Spatial region");
    trackedSpatialOperation = mapping.lookupOrNull(entity->operation);
    if (!trackedSpatialOperation)
      return invalid("tracked Spatial region was not cloned");
  }
  mlir::Operation *mappedOperation = mapping.lookupOrNull(selected->operation);
  if (!mappedOperation)
    return invalid("selected operation was not cloned");
  mappedOperation->setDiscardableAttr(
      kSpecialMathAccuracyAttrName,
      mlir::StringAttr::get(
          clone->getContext(),
          stringifySpecialMathAccuracyTier(decision.accuracy)));

  llvm::DenseSet<std::uint64_t> locatedOperations;
  for (const StructuredOperationSourceProvenance &provenance :
       sourceProvenance) {
    if (provenance.operation.parent != parent.identity() ||
        provenance.operation.kind != StructuredEntityKind::Operation)
      return invalid("source provenance has the wrong Structured owner");
    if (provenance.sourceFiles.empty() ||
        !llvm::is_sorted(provenance.sourceFiles) ||
        std::adjacent_find(provenance.sourceFiles.begin(),
                           provenance.sourceFiles.end()) !=
            provenance.sourceFiles.end())
      return invalid("source provenance files are not canonical");
    if (!locatedOperations.insert(provenance.operation.ordinal).second)
      return invalid("source provenance duplicates an operation");
    auto parentOperation = parentView->resolve(provenance.operation);
    if (!parentOperation)
      return parentOperation.takeError();
    mlir::Operation *mapped = mapping.lookupOrNull(parentOperation->operation);
    if (!mapped)
      return invalid("source-backed operation was not cloned");
    llvm::SmallVector<mlir::Location> fileLocations;
    fileLocations.reserve(provenance.sourceFiles.size());
    for (const std::string &sourceFile : provenance.sourceFiles)
      fileLocations.push_back(
          mlir::FileLineColLoc::get(clone->getContext(), sourceFile, 0, 0));
    mapped->setLoc(mlir::FusedLoc::get(clone->getContext(), fileLocations));
  }

  llvm::DenseSet<std::uint64_t> trackedOrdinals;
  std::vector<mlir::Block *> trackedBlocks;
  trackedBlocks.reserve(trackedBlockReferences.size());
  for (const StructuredEntityRef &reference : trackedBlockReferences) {
    if (reference.parent != parent.identity() ||
        reference.kind != StructuredEntityKind::Block)
      return invalid("tracked block has the wrong Structured owner");
    if (!trackedOrdinals.insert(reference.ordinal).second)
      return invalid("tracked block is duplicated");
    auto parentBlock = parentView->resolve(reference);
    if (!parentBlock)
      return parentBlock.takeError();
    mlir::Block *mapped = mapping.lookupOrNull(parentBlock->block);
    if (!mapped)
      return invalid("tracked block was not cloned");
    trackedBlocks.push_back(mapped);
  }

  if (mlir::failed(mlir::verify(*clone)))
    return invalid("materialized special-math candidate does not verify");
  auto finalized = finalizeStructuredProgramWithTrackedEntities(
      clone.get(), trackedBlocks,
      trackedSpatialOperation ? llvm::ArrayRef(&trackedSpatialOperation, 1)
                              : llvm::ArrayRef<mlir::Operation *>{});
  if (!finalized)
    return finalized.takeError();
  if (finalized->trackedOperations.size() !=
      static_cast<std::size_t>(trackedSpatialOperation != nullptr))
    return invalid("tracked Spatial region projection changed cardinality");
  return MaterializedAccuracyProjection{std::move(finalized->artifact),
                                        std::move(finalized->trackedBlocks),
                                        finalized->trackedOperations.empty()
                                            ? std::nullopt
                                            : std::optional(
                                                  finalized->trackedOperations.front()),
                                        std::move(finalized->sourceProvenance)};
}

} // namespace

llvm::ArrayRef<std::uint8_t>
structuredSpecialMathAccuracyDecisionSchemaBytes() {
  return {reinterpret_cast<const std::uint8_t *>(decisionSchema.data()),
          decisionSchema.size()};
}

llvm::Expected<std::vector<std::uint8_t>>
encodeStructuredSpecialMathAccuracyDecision(
    const StructuredSpecialMathAccuracyDecision &decision) {
  if (decision.operation.kind != StructuredEntityKind::Operation)
    return invalid("decision does not reference an operation");
  std::vector<std::uint8_t> bytes =
      encodeStructuredEntityRef(decision.operation);
  auto tier = encodeSpecialMathAccuracyTier(decision.accuracy);
  if (!tier)
    return tier.takeError();
  bytes.insert(bytes.end(), tier->bytes().begin(), tier->bytes().end());
  return bytes;
}

llvm::Expected<StructuredSpecialMathAccuracyDecision>
adoptStructuredSpecialMathAccuracyDecision(
    llvm::ArrayRef<std::uint8_t> canonicalBytes) {
  if (canonicalBytes.size() <= structuredEntityRefWireSize)
    return invalid("decision payload is truncated");
  auto operation = decodeStructuredEntityRef(
      canonicalBytes.take_front(structuredEntityRefWireSize));
  if (!operation)
    return operation.takeError();
  auto tier = decodeSpecialMathAccuracyTier(
      canonicalBytes.drop_front(structuredEntityRefWireSize));
  if (!tier)
    return tier.takeError();
  StructuredSpecialMathAccuracyDecision decision{*operation, *tier};
  auto reencoded = encodeStructuredSpecialMathAccuracyDecision(decision);
  if (!reencoded)
    return reencoded.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*reencoded) != canonicalBytes)
    return invalid("decision payload does not re-encode exactly");
  return decision;
}

llvm::Expected<std::vector<StructuredSpecialMathAccuracyDecision>>
enumerateStructuredSpecialMathAccuracyDecisions(
    const StructuredProgramCandidate &parent) {
  auto view = parent.view();
  if (!view)
    return view.takeError();
  for (const StructuredEntity &entity :
       view->entities(StructuredEntityKind::Operation)) {
    mlir::Operation *operation = entity.operation;
    if (!operation || !isSelectedSpatialOperation(operation))
      continue;
    std::optional<dataflow::OperationSchemaId> schema =
        dataflow::operationSchemaOf(operation);
    const bool special =
        schema && dataflow::semanticsCase(*schema) ==
                      dataflow::OperationSemanticsCase::SpecialMathAccuracy;
    if (!special) {
      if (operation->getDiscardableAttr(kSpecialMathAccuracyAttrName))
        return invalid("non-special operation carries special-math accuracy");
      continue;
    }
    auto approximation = permitsApproximation(operation);
    if (!approximation)
      return approximation.takeError();
    auto selected = selectedAccuracy(operation, *approximation);
    if (!selected)
      return selected.takeError();
    if (*selected)
      continue;

    std::vector<StructuredSpecialMathAccuracyDecision> result;
    llvm::ArrayRef<SpecialMathAccuracyTier> tiers =
        *approximation ? specialMathAccuracyTiers()
                       : specialMathAccuracyTiers().take_front(1);
    result.reserve(tiers.size());
    for (SpecialMathAccuracyTier tier : tiers)
      result.push_back({entity.reference, tier});
    return result;
  }
  return std::vector<StructuredSpecialMathAccuracyDecision>{};
}

llvm::Expected<MaterializedStructuredSpecialMathCandidate>
materializeStructuredSpecialMathAccuracyDecision(
    const StructuredProgramCandidate &parent,
    const StructuredSpecialMathAccuracyDecision &decision) {
  auto materialized =
      materializeDecision(parent, decision, {}, std::nullopt, {});
  if (!materialized)
    return materialized.takeError();
  return MaterializedStructuredSpecialMathCandidate{
      std::move(materialized->structuredProgram),
      std::move(materialized->sourceProvenance)};
}

llvm::Expected<MaterializedStructuredOwnershipCandidate>
materializeStructuredSpecialMathAccuracyDecision(
    MaterializedStructuredOwnershipCandidate parent,
    const StructuredSpecialMathAccuracyDecision &decision) {
  std::vector<StructuredEntityRef> trackedBlocks;
  trackedBlocks.reserve(parent.blockActivityLineage.size());
  for (const StructuredBlockActivityLineage &lineage :
       parent.blockActivityLineage)
    trackedBlocks.push_back(lineage.childBlock);
  auto child = materializeDecision(parent.structuredProgram, decision,
                                   trackedBlocks, parent.ownedSpatialRegion,
                                   parent.sourceProvenance);
  if (!child)
    return child.takeError();
  if (child->trackedBlocks.size() != parent.blockActivityLineage.size())
    return invalid("special-math block lineage changed cardinality");
  std::vector<StructuredBlockActivityLineage> childLineage;
  childLineage.reserve(parent.blockActivityLineage.size());
  for (auto [childBlock, parentLineage] :
       llvm::zip_equal(child->trackedBlocks, parent.blockActivityLineage))
    childLineage.push_back({childBlock, parentLineage.parentBlock});
  return MaterializedStructuredOwnershipCandidate{
      std::move(child->structuredProgram),
      std::move(child->trackedSpatialRegion), std::move(childLineage),
      std::move(child->sourceProvenance)};
}

} // namespace loom::frontend
