#include "Frontend/Compilation/StructuredSchedule.h"

#include "Dataflow/IR/OperationSchema.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"
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

struct ActorMultiplicity final {
  mlir::Operation *representative = nullptr;
  std::uint64_t count = 0;
};

llvm::Expected<std::uint64_t>
aggregateUnrollCapacity(mlir::scf::ForOp loop,
                        const FabricCapabilityIndex &fabric) {
  std::map<std::vector<std::uint8_t>, ActorMultiplicity> actors;
  llvm::Error projectionError = llvm::Error::success();
  loop.getRegion().walk([&](mlir::Operation *operation) {
    if (projectionError || !dataflow::operationSchemaOf(operation))
      return mlir::WalkResult::advance();
    auto projection =
        dataflow::projectRegisteredActorSchemaProjectionBytes(operation);
    if (!projection) {
      projectionError = projection.takeError();
      return mlir::WalkResult::interrupt();
    }
    ActorMultiplicity &multiplicity = actors[projection->bytes().vec()];
    if (!multiplicity.representative)
      multiplicity.representative = operation;
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

llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>>
cloneAndResolveLoop(const StructuredProgramCandidate &parent,
                    const StructuredEntityRef &reference,
                    mlir::scf::ForOp &clonedLoop) {
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
  mlir::OwningOpRef<mlir::ModuleOp> clone(
      llvm::cast<mlir::ModuleOp>(parent.module()->clone(mapping)));
  clonedLoop = llvm::dyn_cast_or_null<mlir::scf::ForOp>(
      mapping.lookupOrNull(sourceLoop.getOperation()));
  if (!clonedLoop)
    return invalid("selected loop was not mapped into the private clone");
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

} // namespace

llvm::Expected<std::vector<StructuredScheduleDecision>>
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
        raising::hasProvenIndependentIterations(inner))
      decisions.push_back(
          {entity.reference, StructuredScheduleDecisionKind::Interchange, 0});
  }
  return decisions;
}

llvm::Expected<MaterializedStructuredScheduleCandidate>
materializeStructuredScheduleDecision(
    const StructuredProgramCandidate &parent,
    const StructuredScheduleDecision &decision) {
  mlir::scf::ForOp loop;
  auto clone = cloneAndResolveLoop(parent, decision.loop, loop);
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
  }
  if (mlir::failed(mlir::verify(**clone)))
    return invalid("materialized schedule candidate does not verify");
  auto finalized = finalizeStructuredProgramWithTrackedBlocks(clone->get(), {});
  if (!finalized)
    return finalized.takeError();
  return MaterializedStructuredScheduleCandidate{
      std::move(finalized->artifact), std::move(finalized->sourceProvenance)};
}

} // namespace loom::frontend
