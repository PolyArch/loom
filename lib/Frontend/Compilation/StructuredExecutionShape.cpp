#include "Frontend/Compilation/StructuredExecutionShape.h"

#include "Frontend/IR/LoomOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
    "loom.structured_execution_shape.decision.1.0";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "structured_execution_shape_invalid: " +
                                     message);
}

llvm::Expected<llvm::SmallVector<mlir::LLVM::FMulAddOp>>
selectedFMulAdds(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::LLVM::FMulAddOp> selected;
  llvm::Error error = llvm::Error::success();
  module.walk([&](loom::SpatialRegionOp spatial) {
    spatial.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *operation) {
      if (operation != spatial &&
          llvm::isa<mlir::FunctionOpInterface>(operation))
        return mlir::WalkResult::skip();
      auto fmuladd = llvm::dyn_cast<mlir::LLVM::FMulAddOp>(operation);
      if (!fmuladd)
        return mlir::WalkResult::advance();
      if (!raising::canMaterializeFMulAdd(*fmuladd)) {
        error = invalid("selected Spatial region contains an exactly "
                        "unrepresentable llvm.intr.fmuladd");
        return mlir::WalkResult::interrupt();
      }
      selected.push_back(fmuladd);
      return mlir::WalkResult::advance();
    });
    return error ? mlir::WalkResult::interrupt() : mlir::WalkResult::advance();
  });
  if (error)
    return std::move(error);
  return selected;
}

struct MaterializedShapeProjection final {
  StructuredProgramCandidate structuredProgram;
  std::vector<StructuredEntityRef> trackedBlocks;
  std::optional<StructuredEntityRef> trackedSpatialRegion;
  std::vector<StructuredOperationSourceProvenance> sourceProvenance;
};

llvm::Expected<MaterializedShapeProjection> materializeDecision(
    const StructuredProgramCandidate &parent,
    const StructuredExecutionShapeDecision &decision,
    llvm::ArrayRef<StructuredEntityRef> trackedBlockReferences,
    std::optional<StructuredEntityRef> trackedSpatialRegion,
    llvm::ArrayRef<StructuredOperationSourceProvenance> sourceProvenance) {
  auto encoded = encodeStructuredExecutionShapeDecision(decision);
  if (!encoded)
    return encoded.takeError();
  auto parentView = parent.view();
  if (!parentView)
    return parentView.takeError();

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

  auto selected = selectedFMulAdds(clone.get());
  if (!selected)
    return selected.takeError();
  if (selected->empty())
    return invalid("candidate has no unresolved execution-shape choice");
  for (mlir::LLVM::FMulAddOp operation : *selected)
    if (mlir::failed(
            raising::materializeFMulAdd(*operation, decision.fmuladdShape)))
      return invalid("selected fmuladd ceased to be exactly representable");

  auto residual = selectedFMulAdds(clone.get());
  if (!residual)
    return residual.takeError();
  if (!residual->empty())
    return invalid("materialized candidate retained an unresolved choice");
  if (mlir::failed(mlir::verify(*clone)))
    return invalid("materialized execution-shape candidate does not verify");
  auto finalized = finalizeStructuredProgramWithTrackedEntities(
      clone.get(), trackedBlocks,
      trackedSpatialOperation ? llvm::ArrayRef(&trackedSpatialOperation, 1)
                              : llvm::ArrayRef<mlir::Operation *>{});
  if (!finalized)
    return finalized.takeError();
  if (finalized->trackedOperations.size() !=
      static_cast<std::size_t>(trackedSpatialOperation != nullptr))
    return invalid("tracked Spatial region projection changed cardinality");
  return MaterializedShapeProjection{std::move(finalized->artifact),
                                     std::move(finalized->trackedBlocks),
                                     finalized->trackedOperations.empty()
                                         ? std::nullopt
                                         : std::optional(
                                               finalized->trackedOperations.front()),
                                     std::move(finalized->sourceProvenance)};
}

} // namespace

llvm::ArrayRef<std::uint8_t> structuredExecutionShapeDecisionSchemaBytes() {
  return {reinterpret_cast<const std::uint8_t *>(decisionSchema.data()),
          decisionSchema.size()};
}

llvm::Expected<std::vector<std::uint8_t>>
encodeStructuredExecutionShapeDecision(
    const StructuredExecutionShapeDecision &decision) {
  if (decision.fmuladdShape != raising::FMulAddExecutionShape::Fused &&
      decision.fmuladdShape != raising::FMulAddExecutionShape::Split)
    return invalid("decision has an unknown shape");
  return std::vector<std::uint8_t>{
      static_cast<std::uint8_t>(decision.fmuladdShape)};
}

llvm::Expected<StructuredExecutionShapeDecision>
adoptStructuredExecutionShapeDecision(
    llvm::ArrayRef<std::uint8_t> canonicalBytes) {
  if (canonicalBytes.size() != 1)
    return invalid("decision payload has the wrong size");
  if (canonicalBytes.front() >
      static_cast<std::uint8_t>(raising::FMulAddExecutionShape::Split))
    return invalid("decision payload has an unknown shape");
  StructuredExecutionShapeDecision decision{
      static_cast<raising::FMulAddExecutionShape>(canonicalBytes.front())};
  auto reencoded = encodeStructuredExecutionShapeDecision(decision);
  if (!reencoded)
    return reencoded.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*reencoded) != canonicalBytes)
    return invalid("decision payload does not re-encode exactly");
  return decision;
}

llvm::Expected<std::vector<StructuredExecutionShapeDecision>>
enumerateStructuredExecutionShapeDecisions(
    const StructuredProgramCandidate &parent) {
  auto selected = selectedFMulAdds(parent.module());
  if (!selected)
    return selected.takeError();
  if (selected->empty())
    return std::vector<StructuredExecutionShapeDecision>{};
  return std::vector<StructuredExecutionShapeDecision>{
      {raising::FMulAddExecutionShape::Fused},
      {raising::FMulAddExecutionShape::Split}};
}

llvm::Expected<MaterializedStructuredExecutionShapeCandidate>
materializeStructuredExecutionShapeDecision(
    const StructuredProgramCandidate &parent,
    const StructuredExecutionShapeDecision &decision) {
  auto materialized =
      materializeDecision(parent, decision, {}, std::nullopt, {});
  if (!materialized)
    return materialized.takeError();
  return MaterializedStructuredExecutionShapeCandidate{
      std::move(materialized->structuredProgram),
      std::move(materialized->sourceProvenance)};
}

llvm::Expected<MaterializedStructuredOwnershipCandidate>
materializeStructuredExecutionShapeDecision(
    MaterializedStructuredOwnershipCandidate parent,
    const StructuredExecutionShapeDecision &decision) {
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
    return invalid("execution-shape block lineage changed cardinality");
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
