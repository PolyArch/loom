#include "Frontend/Compilation/StructuredSchedule.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/Compilation/StructuredSpecialMathAccuracy.h"
#include "Frontend/IR/LoomOps.h"
#include "Frontend/Raising/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace loom::frontend {
namespace {

constexpr llvm::StringLiteral decisionSchema =
    "loom.structured_schedule.decision.2.0";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "structured_schedule_invalid: " + message);
}

std::optional<std::uint64_t> staticTripCount(mlir::scf::ForOp loop) {
  std::optional<llvm::APInt> count = loop.getStaticTripCount();
  if (!count || count->getActiveBits() > 64)
    return std::nullopt;
  return count->getZExtValue();
}

std::vector<std::uint64_t> properDivisors(std::uint64_t value) {
  std::vector<std::uint64_t> factors;
  if (value <= 2)
    return factors;
  for (std::uint64_t divisor = 2; divisor <= value / divisor; ++divisor) {
    if (value % divisor != 0)
      continue;
    factors.push_back(divisor);
    const std::uint64_t paired = value / divisor;
    if (paired != divisor && paired != value)
      factors.push_back(paired);
  }
  llvm::sort(factors);
  return factors;
}

bool isDefinedOutside(mlir::Value value, mlir::scf::ForOp loop) {
  mlir::Region *region = value.getParentRegion();
  return !region || !loop.getRegion().isAncestor(region);
}

bool isPerfectAdjacentNest(mlir::scf::ForOp outer, mlir::scf::ForOp &inner) {
  auto operations = outer.getBody()->without_terminator();
  if (!llvm::hasSingleElement(operations))
    return false;
  inner = llvm::dyn_cast<mlir::scf::ForOp>(&*operations.begin());
  if (!inner || !outer.getInitArgs().empty() || !inner.getInitArgs().empty())
    return false;
  return isDefinedOutside(inner.getLowerBound(), outer) &&
         isDefinedOutside(inner.getUpperBound(), outer) &&
         isDefinedOutside(inner.getStep(), outer);
}

bool hasInvariantNestedLoopBounds(mlir::scf::ForOp outer) {
  mlir::WalkResult result = outer.walk([&](mlir::scf::ForOp nested) {
    if (nested == outer)
      return mlir::WalkResult::advance();
    if (!isDefinedOutside(nested.getLowerBound(), outer) ||
        !isDefinedOutside(nested.getUpperBound(), outer) ||
        !isDefinedOutside(nested.getStep(), outer))
      return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  });
  return !result.wasInterrupted();
}

llvm::SmallVector<mlir::scf::ForOp>
rectangularParallelNest(mlir::scf::ForOp root) {
  llvm::SmallVector<mlir::scf::ForOp> result;
  mlir::scf::ForOp current = root;
  while (current) {
    result.push_back(current);

    mlir::scf::ForOp child;
    for (mlir::Operation &operation : current.getBody()->without_terminator()) {
      if (auto nested = llvm::dyn_cast<mlir::scf::ForOp>(&operation)) {
        if (child)
          return {};
        child = nested;
      }
    }
    if (!child)
      break;
    current = child;
  }
  return raising::hasProvenIndependentIterations(result)
             ? result
             : llvm::SmallVector<mlir::scf::ForOp>{};
}

mlir::Value toIndex(mlir::OpBuilder &builder, mlir::Location location,
                    mlir::Value value) {
  if (llvm::isa<mlir::IndexType>(value.getType()))
    return value;
  return mlir::arith::IndexCastOp::create(builder, location,
                                          builder.getIndexType(), value);
}

llvm::Expected<mlir::scf::ForallOp>
applyParallelizeNest(mlir::scf::ForOp root) {
  llvm::SmallVector<mlir::scf::ForOp> nest = rectangularParallelNest(root);
  if (nest.size() < 2)
    return invalid("parallel nest preconditions are not satisfied");

  mlir::OpBuilder builder(root);
  llvm::SmallVector<mlir::OpFoldResult> lowerBounds;
  llvm::SmallVector<mlir::OpFoldResult> upperBounds;
  llvm::SmallVector<mlir::OpFoldResult> steps;
  lowerBounds.reserve(nest.size());
  upperBounds.reserve(nest.size());
  steps.reserve(nest.size());
  for (mlir::scf::ForOp loop : nest) {
    lowerBounds.push_back(
        toIndex(builder, loop.getLoc(), loop.getLowerBound()));
    upperBounds.push_back(
        toIndex(builder, loop.getLoc(), loop.getUpperBound()));
    steps.push_back(toIndex(builder, loop.getLoc(), loop.getStep()));
  }

  mlir::scf::ForallOp parallel = mlir::scf::ForallOp::create(
      builder, root.getLoc(), lowerBounds, upperBounds, steps,
      /*outputs=*/mlir::ValueRange{}, /*mapping=*/std::nullopt);
  mlir::IRMapping mapping;
  builder.setInsertionPointToStart(parallel.getBody());
  for (auto [dimension, loop] : llvm::enumerate(nest)) {
    mlir::Value induction = parallel.getInductionVar(dimension);
    if (induction.getType() != loop.getInductionVar().getType())
      induction = mlir::arith::IndexCastOp::create(
          builder, loop.getLoc(), loop.getInductionVar().getType(), induction);
    mapping.map(loop.getInductionVar(), induction);
  }

  auto cloneBody = [&](auto &&self, std::size_t depth) -> void {
    mlir::Operation *child =
        depth + 1 < nest.size() ? nest[depth + 1].getOperation() : nullptr;
    for (mlir::Operation &operation :
         nest[depth].getBody()->without_terminator()) {
      if (&operation == child) {
        self(self, depth + 1);
        continue;
      }
      builder.clone(operation, mapping);
    }
  };
  cloneBody(cloneBody, 0);
  root.erase();
  return parallel;
}

struct ActorMultiplicity final {
  mlir::Operation *representative = nullptr;
  std::uint64_t count = 0;
  std::optional<std::uint64_t> resourceUpperBound;
};

struct AggregateUnrollActorProjection final {
  CanonicalSemanticBytes key;
  std::optional<std::uint64_t> resourceUpperBound;
};

llvm::Expected<AggregateUnrollActorProjection>
projectAggregateUnrollActor(mlir::Operation *operation,
                            const FabricCapabilityIndex &fabric) {
  const std::optional<dataflow::OperationSchemaId> schema =
      dataflow::operationSchemaOf(operation);
  const bool unresolvedSpecialMath =
      schema &&
      dataflow::semanticsCase(*schema) ==
          dataflow::OperationSemanticsCase::SpecialMathAccuracy &&
      !operation->getDiscardableAttr(kSpecialMathAccuracyAttrName);
  if (!unresolvedSpecialMath) {
    auto key = dataflow::projectRegisteredActorSchemaProjectionBytes(operation);
    if (!key)
      return key.takeError();
    return AggregateUnrollActorProjection{std::move(*key), std::nullopt};
  }

  auto projections = projectStructuredSpecialMathAccuracyDomain(operation);
  if (!projections)
    return projections.takeError();
  if (projections->empty())
    return invalid("unresolved special-math domain is empty");
  auto key =
      dataflow::encodeCanonicalActorSchemaProjection(projections->front());
  if (!key)
    return key.takeError();
  auto indexBitWidth = getIndexBitWidth(operation);
  if (!indexBitWidth)
    return indexBitWidth.takeError();
  std::uint64_t resourceUpperBound = 0;
  for (const dataflow::CanonicalActorSchemaProjection &projection :
       *projections) {
    auto count =
        fabric.admittingOperationResourceCount(projection, *indexBitWidth);
    if (!count)
      return count.takeError();
    resourceUpperBound = std::max(resourceUpperBound, *count);
  }
  return AggregateUnrollActorProjection{std::move(*key), resourceUpperBound};
}

llvm::Expected<std::uint64_t>
aggregateUnrollCapacity(mlir::scf::ForOp loop,
                        const FabricCapabilityIndex &fabric) {
  std::map<std::vector<std::uint8_t>, ActorMultiplicity> actors;
  llvm::Error projectionError = llvm::Error::success();
  loop.getRegion().walk([&](mlir::Operation *operation) {
    if (projectionError || !dataflow::operationSchemaOf(operation))
      return mlir::WalkResult::advance();
    auto projection = projectAggregateUnrollActor(operation, fabric);
    if (!projection) {
      projectionError = projection.takeError();
      return mlir::WalkResult::interrupt();
    }
    ActorMultiplicity &multiplicity = actors[projection->key.bytes().vec()];
    if (!multiplicity.representative) {
      multiplicity.representative = operation;
      multiplicity.resourceUpperBound = projection->resourceUpperBound;
    } else if (multiplicity.resourceUpperBound !=
               projection->resourceUpperBound) {
      projectionError = invalid("actor-equivalent capacity bounds disagree");
      return mlir::WalkResult::interrupt();
    }
    const std::optional<std::uint64_t> next =
        llvm::checkedAddUnsigned(multiplicity.count, std::uint64_t{1});
    if (!next) {
      projectionError = invalid("actor multiplicity overflow");
      return mlir::WalkResult::interrupt();
    }
    multiplicity.count = *next;
    return mlir::WalkResult::advance();
  });
  if (projectionError)
    return std::move(projectionError);
  if (actors.empty())
    return std::uint64_t{0};

  std::uint64_t capacity = std::numeric_limits<std::uint64_t>::max();
  for (const auto &entry : actors) {
    mlir::Operation *actor = entry.second.representative;
    auto kind = dataflow::classifyCanonicalDataflowActor(actor);
    if (!kind)
      return invalid("registered actor lost its canonical kind");
    if (entry.second.resourceUpperBound) {
      capacity = std::min(capacity, *entry.second.resourceUpperBound /
                                        entry.second.count);
      continue;
    }
    llvm::Expected<std::uint64_t> resources =
        *kind == dataflow::CanonicalDataflowActorKind::Memory
            ? fabric.admittingMemoryResourceCount(actor)
            : fabric.admittingOperationResourceCount(actor);
    if (!resources)
      return resources.takeError();
    capacity = std::min(capacity, *resources / entry.second.count);
  }
  return capacity;
}

llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> cloneAndResolveLoop(
    const StructuredProgramCandidate &parent,
    const StructuredEntityRef &reference,
    std::optional<StructuredEntityRef> trackedSpatialRegion,
    llvm::ArrayRef<StructuredOperationSourceProvenance> sourceProvenance,
    mlir::scf::ForOp &clonedLoop, mlir::Operation *&clonedSpatialRegion) {
  if (reference.kind != StructuredEntityKind::Operation)
    return invalid("schedule decision does not reference an operation");
  auto view = parent.view();
  if (!view)
    return view.takeError();
  auto entity = view->resolve(reference);
  if (!entity)
    return entity.takeError();
  auto sourceLoop = llvm::dyn_cast_or_null<mlir::scf::ForOp>(entity->operation);
  if (!sourceLoop)
    return invalid("schedule decision does not reference scf.for");

  mlir::IRMapping mapping;
  auto privateClone = cloneStructuredProgramWithSourceLocations(
      parent, sourceProvenance, mapping);
  if (!privateClone)
    return privateClone.takeError();
  mlir::OwningOpRef<mlir::ModuleOp> clone = std::move(*privateClone);
  clonedLoop = llvm::dyn_cast_or_null<mlir::scf::ForOp>(
      mapping.lookupOrNull(sourceLoop.getOperation()));
  if (!clonedLoop)
    return invalid("selected loop was not mapped into the private clone");
  if (trackedSpatialRegion) {
    auto spatialEntity = view->resolve(*trackedSpatialRegion);
    if (!spatialEntity)
      return spatialEntity.takeError();
    if (!llvm::isa_and_nonnull<loom::SpatialRegionOp>(spatialEntity->operation))
      return invalid("tracked operation is not a Spatial region");
    clonedSpatialRegion = mapping.lookupOrNull(spatialEntity->operation);
    if (!clonedSpatialRegion)
      return invalid("tracked Spatial region was not mapped into the clone");
  }
  return clone;
}

llvm::Error applyTile(mlir::scf::ForOp loop, std::uint64_t factor) {
  if (factor <= 1 || !loop.getInitArgs().empty())
    return invalid("tile factor or loop shape is not canonical");
  mlir::OpBuilder builder(loop);
  mlir::Value size = mlir::arith::ConstantOp::create(
      builder, loop.getLoc(),
      builder.getIntegerAttr(loop.getStep().getType(), factor));
  if (mlir::tilePerfectlyNested(loop, {size}).empty())
    return invalid("SCF tiling did not materialize an intra-tile loop");
  return llvm::Error::success();
}

llvm::Error applyUnroll(mlir::scf::ForOp loop, std::uint64_t factor) {
  if (factor <= 1 || !loop.getInitArgs().empty())
    return invalid("unroll factor or loop shape is not canonical");
  if (mlir::failed(mlir::loopUnrollByFactor(loop, factor)))
    return invalid("SCF unroll rejected the selected decision");
  return llvm::Error::success();
}

llvm::Error applyInterchange(mlir::scf::ForOp outer) {
  mlir::scf::ForOp inner;
  if (!isPerfectAdjacentNest(outer, inner) ||
      !raising::hasProvenIndependentIterations(outer) ||
      !raising::hasProvenIndependentIterations(inner))
    return invalid("interchange preconditions are not satisfied");

  mlir::OpBuilder builder(outer);
  mlir::scf::ForOp newOuter = mlir::scf::ForOp::create(
      builder, inner.getLoc(), inner.getLowerBound(), inner.getUpperBound(),
      inner.getStep(), mlir::ValueRange{}, nullptr, inner.getUnsignedCmp());
  newOuter->setAttrs(inner->getAttrs());
  builder.setInsertionPoint(newOuter.getBody()->getTerminator());
  mlir::scf::ForOp newInner = mlir::scf::ForOp::create(
      builder, outer.getLoc(), outer.getLowerBound(), outer.getUpperBound(),
      outer.getStep(), mlir::ValueRange{}, nullptr, outer.getUnsignedCmp());
  newInner->setAttrs(outer->getAttrs());

  mlir::IRMapping mapping;
  mapping.map(outer.getInductionVar(), newInner.getInductionVar());
  mapping.map(inner.getInductionVar(), newOuter.getInductionVar());
  builder.setInsertionPoint(newInner.getBody()->getTerminator());
  for (mlir::Operation &operation : inner.getBody()->without_terminator())
    builder.clone(operation, mapping);
  outer.erase();
  return llvm::Error::success();
}

llvm::Error applyUnrollAndJam(mlir::scf::ForOp loop, std::uint64_t factor) {
  if (factor <= 1 || !loop.getInitArgs().empty())
    return invalid("unroll-and-jam factor or loop shape is not canonical");
  if (mlir::failed(mlir::loopUnrollJamByFactor(loop, factor)))
    return invalid("SCF unroll-and-jam rejected the selected decision");
  return llvm::Error::success();
}

} // namespace

llvm::ArrayRef<std::uint8_t> structuredScheduleDecisionSchemaBytes() {
  return {reinterpret_cast<const std::uint8_t *>(decisionSchema.data()),
          decisionSchema.size()};
}

llvm::Expected<std::vector<std::uint8_t>>
encodeStructuredScheduleDecision(const StructuredScheduleDecision &decision) {
  if (decision.loop.kind != StructuredEntityKind::Operation)
    return invalid("decision does not reference an operation");
  if (static_cast<std::uint32_t>(decision.kind) >
      static_cast<std::uint32_t>(
          StructuredScheduleDecisionKind::ParallelizeNest))
    return invalid("decision has an unknown kind");
  const bool factorless =
      decision.kind == StructuredScheduleDecisionKind::Interchange ||
      decision.kind == StructuredScheduleDecisionKind::Parallelize ||
      decision.kind == StructuredScheduleDecisionKind::ParallelizeNest;
  if ((factorless && decision.factor != 0) ||
      (!factorless && decision.factor <= 1))
    return invalid("decision has an invalid factor");
  std::vector<std::uint8_t> bytes = encodeStructuredEntityRef(decision.loop);
  const auto appendU32 = [&](std::uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8)
      bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  };
  const auto appendU64 = [&](std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
      bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  };
  appendU32(static_cast<std::uint32_t>(decision.kind));
  appendU64(decision.factor);
  return bytes;
}

llvm::Expected<StructuredScheduleDecision>
adoptStructuredScheduleDecision(llvm::ArrayRef<std::uint8_t> canonicalBytes) {
  constexpr std::size_t wireSize = structuredEntityRefWireSize + 12;
  if (canonicalBytes.size() != wireSize)
    return invalid("decision payload has the wrong size");
  auto loop = decodeStructuredEntityRef(
      canonicalBytes.take_front(structuredEntityRefWireSize));
  if (!loop)
    return loop.takeError();
  if (loop->kind != StructuredEntityKind::Operation)
    return invalid("decision does not reference an operation");
  llvm::ArrayRef<std::uint8_t> suffix =
      canonicalBytes.drop_front(structuredEntityRefWireSize);
  std::uint32_t kind = 0;
  for (std::uint8_t byte : suffix.take_front(4))
    kind = (kind << 8) | byte;
  if (kind > static_cast<std::uint32_t>(
                 StructuredScheduleDecisionKind::ParallelizeNest))
    return invalid("decision payload has an unknown kind");
  std::uint64_t factor = 0;
  for (std::uint8_t byte : suffix.drop_front(4))
    factor = (factor << 8) | byte;
  const auto typedKind = static_cast<StructuredScheduleDecisionKind>(kind);
  const bool factorless =
      typedKind == StructuredScheduleDecisionKind::Interchange ||
      typedKind == StructuredScheduleDecisionKind::Parallelize ||
      typedKind == StructuredScheduleDecisionKind::ParallelizeNest;
  if ((factorless && factor != 0) || (!factorless && factor <= 1))
    return invalid("decision payload has an invalid factor");
  StructuredScheduleDecision decision{*loop, typedKind, factor};
  auto reencoded = encodeStructuredScheduleDecision(decision);
  if (!reencoded)
    return reencoded.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*reencoded) != canonicalBytes)
    return invalid("decision payload does not re-encode exactly");
  return decision;
}

llvm::Expected<StructuredScheduleDecisionDomain>
enumerateStructuredScheduleDecisions(const StructuredProgramCandidate &parent,
                                     const fabric::FinalizedFabricRoot &fabric,
                                     std::uint64_t scopeExpansionLimit) {
  if (scopeExpansionLimit == 0)
    return invalid("scope expansion limit must be positive");
  auto view = parent.view();
  if (!view)
    return view.takeError();
  FabricCapabilityIndex capabilityIndex(fabric.view());
  std::vector<StructuredScheduleDecision> decisions;
  std::uint64_t expanded = 0;
  for (const StructuredEntity &entity :
       view->entities(StructuredEntityKind::Operation)) {
    auto loop = llvm::dyn_cast_or_null<mlir::scf::ForOp>(entity.operation);
    if (!loop)
      continue;
    if (expanded == scopeExpansionLimit)
      break;
    ++expanded;

    std::optional<std::uint64_t> tripCount = staticTripCount(loop);
    if (tripCount && *tripCount > 1 && loop.getInitArgs().empty()) {
      std::vector<std::uint64_t> factors = properDivisors(*tripCount);
      for (std::uint64_t factor : factors)
        decisions.push_back(
            {entity.reference, StructuredScheduleDecisionKind::Tile, factor});

      auto capacity = aggregateUnrollCapacity(loop, capabilityIndex);
      if (!capacity)
        return capacity.takeError();
      for (std::uint64_t factor : factors) {
        if (factor > *capacity)
          break;
        decisions.push_back(
            {entity.reference, StructuredScheduleDecisionKind::Unroll, factor});
      }
    }

    mlir::scf::ForOp inner;
    if (isPerfectAdjacentNest(loop, inner) &&
        raising::hasProvenIndependentIterations(loop) &&
        raising::hasProvenIndependentIterations(inner)) {
      decisions.push_back(
          {entity.reference, StructuredScheduleDecisionKind::Interchange, 0});
      if (tripCount && *tripCount > 1 && hasInvariantNestedLoopBounds(loop)) {
        auto capacity = aggregateUnrollCapacity(loop, capabilityIndex);
        if (!capacity)
          return capacity.takeError();
        for (std::uint64_t factor : properDivisors(*tripCount)) {
          if (factor > *capacity)
            break;
          decisions.push_back({entity.reference,
                               StructuredScheduleDecisionKind::UnrollAndJam,
                               factor});
        }
      }
    }
    if (loop.getInitArgs().empty() &&
        raising::hasProvenIndependentIterations(loop))
      decisions.push_back(
          {entity.reference, StructuredScheduleDecisionKind::Parallelize, 0});
    if (rectangularParallelNest(loop).size() >= 2)
      decisions.push_back({entity.reference,
                           StructuredScheduleDecisionKind::ParallelizeNest, 0});
  }
  return StructuredScheduleDecisionDomain{std::move(decisions), expanded};
}

llvm::Expected<MaterializedStructuredScheduleCandidate>
materializeStructuredScheduleDecision(
    const StructuredProgramCandidate &parent,
    const StructuredScheduleDecision &decision,
    std::optional<StructuredEntityRef> trackedSpatialRegion,
    llvm::ArrayRef<StructuredOperationSourceProvenance> sourceProvenance) {
  auto encoded = encodeStructuredScheduleDecision(decision);
  if (!encoded)
    return encoded.takeError();
  mlir::scf::ForOp loop;
  mlir::Operation *clonedSpatialRegion = nullptr;
  auto clone = cloneAndResolveLoop(parent, decision.loop, trackedSpatialRegion,
                                   sourceProvenance, loop, clonedSpatialRegion);
  if (!clone)
    return clone.takeError();

  switch (decision.kind) {
  case StructuredScheduleDecisionKind::Tile:
    if (llvm::Error error = applyTile(loop, decision.factor))
      return std::move(error);
    break;
  case StructuredScheduleDecisionKind::Unroll:
    if (llvm::Error error = applyUnroll(loop, decision.factor))
      return std::move(error);
    break;
  case StructuredScheduleDecisionKind::Interchange:
    if (decision.factor != 0)
      return invalid("interchange decision carries a factor");
    if (llvm::Error error = applyInterchange(loop))
      return std::move(error);
    break;
  case StructuredScheduleDecisionKind::UnrollAndJam:
    if (llvm::Error error = applyUnrollAndJam(loop, decision.factor))
      return std::move(error);
    break;
  case StructuredScheduleDecisionKind::Parallelize:
    if (decision.factor != 0)
      return invalid("parallelize decision carries a factor");
    if (mlir::failed(raising::materializeIndependentLoopAsForall(loop)))
      return invalid("SCF parallelization rejected the selected decision");
    break;
  case StructuredScheduleDecisionKind::ParallelizeNest:
    if (decision.factor != 0)
      return invalid("parallel-nest decision carries a factor");
    if (auto parallel = applyParallelizeNest(loop)) {
      const bool insideTrackedSpatial =
          clonedSpatialRegion &&
          clonedSpatialRegion->isAncestor(parallel->getOperation());
      if (insideTrackedSpatial)
        if (llvm::Error error =
                materializeOwnedSpatialForallThreadDomain(*parallel))
          return std::move(error);
    } else {
      return parallel.takeError();
    }
    break;
  }
  if (mlir::failed(mlir::verify(**clone)))
    return invalid("materialized schedule candidate does not verify");
  auto finalized = finalizeStructuredProgramWithTrackedEntities(
      clone->get(), {},
      clonedSpatialRegion ? llvm::ArrayRef(&clonedSpatialRegion, 1)
                          : llvm::ArrayRef<mlir::Operation *>{});
  if (!finalized)
    return finalized.takeError();
  if (finalized->trackedOperations.size() !=
      static_cast<std::size_t>(clonedSpatialRegion != nullptr))
    return invalid("tracked Spatial region projection changed cardinality");
  return MaterializedStructuredScheduleCandidate{
      std::move(finalized->artifact),
      finalized->trackedOperations.empty()
          ? std::nullopt
          : std::optional(finalized->trackedOperations.front()),
      std::move(finalized->sourceProvenance)};
}

} // namespace loom::frontend
