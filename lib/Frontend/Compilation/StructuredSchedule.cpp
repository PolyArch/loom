#include "Frontend/Compilation/StructuredSchedule.h"

#include "StructuredPolyhedralMaterializer.h"
#include "StructuredScheduleInternal.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/Compilation/StructuredSpecialMathAccuracy.h"
#include "Frontend/IR/LoomOps.h"
#include "Frontend/Lowering/GraphParallelLowering.h"
#include "Frontend/Raising/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom::frontend {
char StructuredScheduleProposalRefusal::ID = 0;

std::string StructuredScheduleProposalRefusal::message() const {
  return "structured schedule proposal refused with kind " +
         std::to_string(static_cast<std::uint32_t>(kind_));
}

void StructuredScheduleProposalRefusal::log(llvm::raw_ostream &stream) const {
  stream << "structured_schedule_proposal_refused: loop=" << loop_.ordinal
         << " kind=" << static_cast<std::uint32_t>(kind_);
}

std::error_code StructuredScheduleProposalRefusal::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

namespace detail {

llvm::Error validateStructuredVectorScheduleCoordinate(
    const StructuredVectorScheduleCoordinate &coordinate) {
  const auto invalidCoordinate = [](const llvm::Twine &message) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "structured_schedule_invalid: " + message);
  };
  if (coordinate.shape.size() != 1 || coordinate.shape.front() <= 1 ||
      coordinate.shape.front() > maximumCanonicalStructuredScheduleFactor)
    return invalidCoordinate(
        "vector coordinate has no supported rank-one shape");
  if (coordinate.requiredAlignmentBytes == 0)
    return invalidCoordinate("vector coordinate has no required alignment");
  if (coordinate.tailPolicy > StructuredVectorTailPolicy::ReductionMask ||
      coordinate.aliasPolicy !=
          StructuredVectorAliasPolicy::ProviderProvenNoAlias ||
      coordinate.reductionSchedule >
          StructuredReductionSchedule::FloatingReassociated)
    return invalidCoordinate("vector coordinate has an unknown typed policy");
  if (coordinate.tailPolicy == StructuredVectorTailPolicy::ReductionMask &&
      coordinate.reductionSchedule == StructuredReductionSchedule::None)
    return invalidCoordinate(
        "non-reduction vector coordinate selects a reduction mask");
  return llvm::Error::success();
}

} // namespace detail

namespace {

constexpr llvm::StringLiteral decisionSchema =
    "loom.structured_schedule.decision.5.0";

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

std::vector<std::uint64_t> canonicalProperDivisors(std::uint64_t value) {
  std::vector<std::uint64_t> factors;
  if (value <= 2)
    return factors;
  const std::uint64_t maximumFactor =
      std::min(maximumCanonicalStructuredScheduleFactor, value - 1);
  for (std::uint64_t factor = 2; factor <= maximumFactor; ++factor)
    if (value % factor == 0)
      factors.push_back(factor);
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

llvm::SmallVector<mlir::scf::ForOp> rectangularNest(mlir::scf::ForOp root) {
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
  return result;
}

llvm::SmallVector<mlir::scf::ForOp>
rectangularParallelNest(mlir::scf::ForOp root) {
  llvm::SmallVector<mlir::scf::ForOp> result = rectangularNest(root);
  return raising::proveIndependentIterations(result) ==
                 raising::ParallelDependenceResult::ProvenIndependent
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
    mlir::IRMapping &mapping, mlir::Operation *&clonedLoop,
    mlir::Operation *&clonedSpatialRegion) {
  if (reference.kind != StructuredEntityKind::Operation)
    return invalid("schedule decision does not reference an operation");
  auto view = parent.view();
  if (!view)
    return view.takeError();
  auto entity = view->resolve(reference);
  if (!entity)
    return entity.takeError();
  mlir::Operation *sourceLoop = entity->operation;
  if (!llvm::isa_and_nonnull<mlir::scf::ForOp, mlir::affine::AffineForOp>(
          sourceLoop))
    return invalid("schedule decision does not reference a supported loop");

  auto privateClone = cloneStructuredProgramWithSourceLocations(
      parent, sourceProvenance, mapping);
  if (!privateClone)
    return privateClone.takeError();
  mlir::OwningOpRef<mlir::ModuleOp> clone = std::move(*privateClone);
  clonedLoop = mapping.lookupOrNull(sourceLoop);
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

llvm::Error interchangeScfLoops(mlir::scf::ForOp outer,
                                mlir::scf::ForOp inner) {
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

llvm::Error applyInterchange(mlir::scf::ForOp outer) {
  mlir::scf::ForOp inner;
  if (!isPerfectAdjacentNest(outer, inner) ||
      raising::proveIndependentIterations(outer) !=
          raising::ParallelDependenceResult::ProvenIndependent ||
      raising::proveIndependentIterations(inner) !=
          raising::ParallelDependenceResult::ProvenIndependent)
    return invalid("interchange preconditions are not satisfied");

  return interchangeScfLoops(outer, inner);
}

bool isStructuredLoop(mlir::Operation *operation) {
  return llvm::isa<mlir::scf::ForOp, mlir::affine::AffineForOp>(operation);
}

mlir::Block *structuredLoopBody(mlir::Operation *operation) {
  if (auto loop = llvm::dyn_cast<mlir::scf::ForOp>(operation))
    return loop.getBody();
  if (auto loop = llvm::dyn_cast<mlir::affine::AffineForOp>(operation))
    return loop.getBody();
  return nullptr;
}

llvm::SmallVector<mlir::Operation *>
perfectStructuredNest(mlir::Operation *root) {
  llvm::SmallVector<mlir::Operation *> loops;
  mlir::Operation *current = root;
  while (isStructuredLoop(current)) {
    loops.push_back(current);
    mlir::Operation *nested = nullptr;
    bool hasDirectStatement = false;
    for (mlir::Operation &operation :
         structuredLoopBody(current)->without_terminator()) {
      if (isStructuredLoop(&operation)) {
        if (nested || hasDirectStatement)
          return {};
        nested = &operation;
      } else {
        if (nested)
          return {};
        hasDirectStatement = true;
      }
    }
    if (!nested)
      break;
    current = nested;
  }
  return loops;
}

bool valueDefinedOutside(mlir::Value value, mlir::Operation *root) {
  mlir::Region *region = value.getParentRegion();
  return !region || !root->getRegion(0).isAncestor(region);
}

bool hasMaterializableAdjacentInterchange(mlir::Operation *root) {
  llvm::SmallVector<mlir::Operation *> nest = perfectStructuredNest(root);
  if (nest.size() < 2)
    return false;
  if (auto outer = llvm::dyn_cast<mlir::scf::ForOp>(nest[0])) {
    auto inner = llvm::dyn_cast<mlir::scf::ForOp>(nest[1]);
    return inner && outer.getInitArgs().empty() &&
           inner.getInitArgs().empty() &&
           isDefinedOutside(inner.getLowerBound(), outer) &&
           isDefinedOutside(inner.getUpperBound(), outer) &&
           isDefinedOutside(inner.getStep(), outer);
  }
  auto outer = llvm::dyn_cast<mlir::affine::AffineForOp>(nest[0]);
  auto inner = llvm::dyn_cast<mlir::affine::AffineForOp>(nest[1]);
  if (!outer || !inner || !outer.getInits().empty() ||
      !inner.getInits().empty())
    return false;
  return llvm::all_of(inner.getLowerBoundOperands(),
                      [&](mlir::Value value) {
                        return valueDefinedOutside(value, outer);
                      }) &&
         llvm::all_of(inner.getUpperBoundOperands(), [&](mlir::Value value) {
           return valueDefinedOutside(value, outer);
         });
}

bool scheduleFormDistributesStatements(StructuredPolyhedralScheduleForm form) {
  return form == StructuredPolyhedralScheduleForm::StatementMajor ||
         form == StructuredPolyhedralScheduleForm::
                     StatementMajorAdjacentInterchange;
}

bool scheduleFormInterchangesDimensions(StructuredPolyhedralScheduleForm form) {
  return form == StructuredPolyhedralScheduleForm::AdjacentInterchange ||
         form == StructuredPolyhedralScheduleForm::
                     StatementMajorAdjacentInterchange;
}

bool canMaterializeCanonicalPolyhedralSchedule(
    mlir::Operation *root, const StructuredPolyhedralScopView &scop) {
  if (scop.imperfectNest || scop.loopCount == 0 ||
      scop.loopCount != scop.maximumLoopDepth ||
      scop.schedule.form == StructuredPolyhedralScheduleForm::General)
    return false;
  llvm::SmallVector<mlir::Operation *> nest = perfectStructuredNest(root);
  if (nest.size() != scop.loopCount)
    return false;
  mlir::Block *innermost = structuredLoopBody(nest.back());
  if (!innermost || static_cast<std::size_t>(
                        std::distance(innermost->without_terminator().begin(),
                                      innermost->without_terminator().end())) !=
                        scop.statements.size())
    return false;
  if (scheduleFormDistributesStatements(scop.schedule.form) &&
      llvm::any_of(scop.dependences, [](const auto &dependence) {
        return dependence.kind == StructuredPolyhedralDependenceKind::ScalarSsa;
      }))
    return false;
  return !scheduleFormInterchangesDimensions(scop.schedule.form) ||
         hasMaterializableAdjacentInterchange(root);
}

llvm::Expected<std::optional<StructuredScopRefusalKind>>
classifyPolyhedralScheduleMaterialization(
    mlir::Operation *root, const StructuredPolyhedralScopView &scop) {
  if (scop.schedule.form == StructuredPolyhedralScheduleForm::SourceOrder)
    return std::nullopt;
  if (canMaterializeCanonicalPolyhedralSchedule(root, scop))
    return std::nullopt;
  return detail::classifyPinnedIslScheduleMaterialization(root, scop);
}

llvm::Error applyProvenAdjacentInterchange(mlir::Operation *root) {
  if (!hasMaterializableAdjacentInterchange(root))
    return invalid("polyhedral interchange shape is not materializable");
  llvm::SmallVector<mlir::Operation *> nest = perfectStructuredNest(root);
  if (auto outer = llvm::dyn_cast<mlir::scf::ForOp>(nest[0]))
    return interchangeScfLoops(outer, llvm::cast<mlir::scf::ForOp>(nest[1]));
  mlir::affine::interchangeLoops(
      llvm::cast<mlir::affine::AffineForOp>(nest[0]),
      llvm::cast<mlir::affine::AffineForOp>(nest[1]));
  return llvm::Error::success();
}

llvm::Expected<llvm::SmallVector<mlir::Operation *>>
distributePolyhedralStatements(mlir::Operation *root,
                               std::size_t statementCount) {
  llvm::SmallVector<mlir::Operation *> nest = perfectStructuredNest(root);
  if (nest.empty())
    return invalid("polyhedral distribution source is not a perfect nest");
  mlir::Block *innermost = structuredLoopBody(nest.back());
  llvm::SmallVector<mlir::Operation *> statements;
  for (mlir::Operation &operation : innermost->without_terminator())
    statements.push_back(&operation);
  if (statements.size() != statementCount)
    return invalid("polyhedral distribution statement count changed");

  llvm::SmallVector<mlir::Operation *> roots;
  roots.reserve(statementCount);
  mlir::OpBuilder builder(root);
  for (mlir::Operation *statement : statements) {
    builder.setInsertionPoint(root);
    mlir::IRMapping mapping;
    mlir::Operation *clone = builder.clone(*root, mapping);
    mlir::Operation *selected = mapping.lookupOrNull(statement);
    llvm::SmallVector<mlir::Operation *> clonedNest =
        perfectStructuredNest(clone);
    if (!selected || clonedNest.size() != nest.size())
      return invalid("polyhedral distribution clone lost its statement");
    mlir::Block *clonedBody = structuredLoopBody(clonedNest.back());
    if (selected->getBlock() != clonedBody)
      return invalid("polyhedral distribution changed statement nesting");
    llvm::SmallVector<mlir::Operation *> discarded;
    for (mlir::Operation &operation : clonedBody->without_terminator())
      if (&operation != selected)
        discarded.push_back(&operation);
    for (mlir::Operation *operation : llvm::reverse(discarded))
      operation->erase();
    roots.push_back(clone);
  }
  root->erase();
  return roots;
}

llvm::Error applyPolyhedralSchedule(mlir::Operation *root,
                                    const StructuredPolyhedralScopView &scop) {
  if (!canMaterializeCanonicalPolyhedralSchedule(root, scop) ||
      scop.schedule.form == StructuredPolyhedralScheduleForm::SourceOrder)
    return invalid("polyhedral schedule form is not a transform coordinate");
  if (!scheduleFormDistributesStatements(scop.schedule.form))
    return applyProvenAdjacentInterchange(root);
  auto roots = distributePolyhedralStatements(root, scop.statements.size());
  if (!roots)
    return roots.takeError();
  if (scheduleFormInterchangesDimensions(scop.schedule.form))
    for (mlir::Operation *distributedRoot : *roots)
      if (llvm::Error error = applyProvenAdjacentInterchange(distributedRoot))
        return error;
  return llvm::Error::success();
}

llvm::Error applyUnrollAndJam(mlir::scf::ForOp loop, std::uint64_t factor) {
  if (factor <= 1 || !loop.getInitArgs().empty())
    return invalid("unroll-and-jam factor or loop shape is not canonical");
  if (mlir::failed(mlir::loopUnrollJamByFactor(loop, factor)))
    return invalid("SCF unroll-and-jam rejected the selected decision");
  return llvm::Error::success();
}

llvm::Expected<
    std::variant<StructuredVectorScheduleCoordinate, StructuredScopRefusalKind>>
coordinateFor(const ExactStructuredScopView &scop, std::uint64_t factor) {
  if (!scop.constantTripCount || *scop.constantTripCount <= 1 ||
      factor > *scop.constantTripCount ||
      factor > maximumCanonicalStructuredScheduleFactor)
    return StructuredScopRefusalKind::NonCanonicalIterationDomain;
  const std::optional<std::uint64_t> requiredAlignment =
      llvm::checkedMulUnsigned(scop.maximumElementBytes, factor);
  if (!requiredAlignment)
    return invalid("vector alignment requirement overflows u64");
  if (llvm::any_of(scop.accesses, [&](const StructuredScopAccessView &access) {
        return access.elementBytes != scop.maximumElementBytes ||
               access.alignmentBytes % *requiredAlignment != 0;
      }))
    return StructuredScopRefusalKind::AlignmentProofNotEstablished;

  StructuredVectorTailPolicy tail = StructuredVectorTailPolicy::Exact;
  const bool divisible = *scop.constantTripCount % factor == 0;
  if (!divisible) {
    if (scop.reductionSchedule == StructuredReductionSchedule::None)
      return StructuredScopRefusalKind::UnsupportedTail;
    tail = StructuredVectorTailPolicy::ReductionMask;
  }
  return StructuredVectorScheduleCoordinate{
      {factor},
      tail,
      *requiredAlignment,
      StructuredVectorAliasPolicy::ProviderProvenNoAlias,
      scop.reductionSchedule};
}

using VectorizationAttempt =
    std::variant<mlir::affine::AffineForOp, StructuredScopRefusalKind>;

llvm::Expected<VectorizationAttempt>
applyVectorize(mlir::affine::AffineForOp loop,
               const StructuredVectorScheduleCoordinate &coordinate) {
  if (llvm::Error error =
          detail::validateStructuredVectorScheduleCoordinate(coordinate))
    return std::move(error);
  llvm::SmallVector<mlir::affine::LoopReduction> reductions;
  if (!mlir::affine::isLoopParallel(loop, &reductions))
    return StructuredScopRefusalKind::ProviderMaterializationRejected;
  if ((coordinate.reductionSchedule == StructuredReductionSchedule::None) !=
      reductions.empty())
    return invalid("vector reduction coordinate differs from the source loop");

  mlir::affine::VectorizationStrategy strategy;
  strategy.vectorSizes.push_back(
      static_cast<std::int64_t>(coordinate.shape.front()));
  strategy.loopToVectorDim[loop.getOperation()] = 0;
  if (!reductions.empty())
    strategy.reductionLoops[loop.getOperation()] = reductions;

  mlir::Block *block = loop->getBlock();
  mlir::Operation *predecessor = loop->getPrevNode();
  mlir::Operation *successor = loop->getNextNode();
  std::vector<llvm::SmallVector<mlir::affine::AffineForOp, 2>> loops = {{loop}};
  if (mlir::failed(mlir::affine::vectorizeAffineLoopNest(loops, strategy)))
    return StructuredScopRefusalKind::ProviderMaterializationRejected;

  mlir::affine::AffineForOp replacement;
  mlir::Operation *operation =
      predecessor ? predecessor->getNextNode() : &block->front();
  while (operation && operation != successor) {
    if (auto candidate = llvm::dyn_cast<mlir::affine::AffineForOp>(operation)) {
      if (replacement)
        return invalid("Affine vectorizer produced an ambiguous loop root");
      replacement = candidate;
    }
    operation = operation->getNextNode();
  }
  if (!replacement || replacement.getStepAsInt() !=
                          static_cast<std::int64_t>(coordinate.shape.front()))
    return invalid("Affine vectorizer did not materialize the selected shape");
  return replacement;
}

llvm::Expected<mlir::vector::CreateMaskOp>
selectedTailMask(mlir::affine::AffineForOp loop) {
  mlir::vector::CreateMaskOp result;
  loop.walk([&](mlir::vector::CreateMaskOp candidate) {
    if (result)
      return mlir::WalkResult::interrupt();
    result = candidate;
    return mlir::WalkResult::advance();
  });
  if (!result)
    return invalid("masked vector coordinate produced no tail mask");
  return result;
}

bool transferResultIsTailGuarded(mlir::vector::TransferReadOp read,
                                 mlir::Value mask) {
  return !read.getResult().use_empty() &&
         llvm::all_of(read.getResult().getUsers(), [&](mlir::Operation *user) {
           auto select = llvm::dyn_cast<mlir::arith::SelectOp>(user);
           return select && select.getCondition() == mask &&
                  select.getTrueValue() == read.getResult();
         });
}

llvm::Error attachTailMask(mlir::affine::AffineForOp loop) {
  auto mask = selectedTailMask(loop);
  if (!mask)
    return mask.takeError();
  llvm::SmallVector<mlir::vector::TransferReadOp> reads;
  llvm::SmallVector<mlir::vector::TransferWriteOp> writes;
  loop.walk([&](mlir::Operation *operation) {
    if (auto read = llvm::dyn_cast<mlir::vector::TransferReadOp>(operation))
      reads.push_back(read);
    else if (auto write =
                 llvm::dyn_cast<mlir::vector::TransferWriteOp>(operation))
      writes.push_back(write);
  });
  for (mlir::vector::TransferReadOp read : reads) {
    if (!mlir::matchPattern(read.getPadding(), mlir::m_Zero()) &&
        !transferResultIsTailGuarded(read, mask->getResult()))
      return invalid("masked vector read has observable nonzero padding");
    mlir::OpBuilder builder(read);
    auto replacement = mlir::vector::TransferReadOp::create(
        builder, read.getLoc(), read.getVectorType(), read.getBase(),
        read.getIndices(), read.getPermutationMapAttr(), read.getPadding(),
        mask->getResult(), builder.getBoolArrayAttr({false}));
    read.getResult().replaceAllUsesWith(replacement.getResult());
    read.erase();
  }
  for (mlir::vector::TransferWriteOp write : writes) {
    mlir::OpBuilder builder(write);
    mlir::vector::TransferWriteOp::create(
        builder, write.getLoc(), write.getValueToStore(), write.getBase(),
        write.getIndices(), write.getPermutationMapAttr(), mask->getResult(),
        builder.getBoolArrayAttr({false}));
    write.erase();
  }
  return llvm::Error::success();
}

llvm::Error lowerTailMask(mlir::affine::AffineForOp loop) {
  auto mask = selectedTailMask(loop);
  if (!mask)
    return mask.takeError();
  if (mask->getNumOperands() != 1)
    return invalid("rank-one tail mask has the wrong bound arity");
  mlir::OpBuilder builder(*mask);
  mlir::VectorType type = mask->getResult().getType();
  auto falseElements =
      mlir::DenseElementsAttr::get(type, builder.getBoolAttr(false));
  mlir::Value result = mlir::arith::ConstantOp::create(builder, mask->getLoc(),
                                                       type, falseElements);
  for (std::int64_t lane = 0; lane != type.getDimSize(0); ++lane) {
    mlir::Value laneValue =
        mlir::arith::ConstantIndexOp::create(builder, mask->getLoc(), lane);
    mlir::Value active = mlir::arith::CmpIOp::create(
        builder, mask->getLoc(), mlir::arith::CmpIPredicate::slt, laneValue,
        mask->getOperand(0));
    result =
        mlir::vector::InsertOp::create(builder, mask->getLoc(), active, result,
                                       llvm::ArrayRef<std::int64_t>{lane});
  }
  mask->getResult().replaceAllUsesWith(result);
  mask->erase();
  return llvm::Error::success();
}

llvm::SmallVector<mlir::Operation *> vectorizedClosure(mlir::Operation *root);

using VectorLoweringAttempt =
    std::variant<mlir::scf::ForOp, StructuredScopRefusalKind>;

llvm::Expected<VectorLoweringAttempt>
lowerVectorizedScop(mlir::affine::AffineForOp loop,
                    const StructuredVectorScheduleCoordinate &coordinate) {
  if (coordinate.tailPolicy == StructuredVectorTailPolicy::ReductionMask) {
    if (llvm::Error error = lowerTailMask(loop))
      return std::move(error);
  }

  mlir::Block *block = loop->getBlock();
  mlir::Operation *predecessor = loop->getPrevNode();
  mlir::Operation *successor = loop->getNextNode();
  mlir::MLIRContext *context = loop.getContext();
  llvm::SmallVector<mlir::vector::ReductionOp> reductions;
  for (mlir::Value result : loop.getResults()) {
    for (mlir::Operation *user : result.getUsers()) {
      if (auto reduction = llvm::dyn_cast<mlir::vector::ReductionOp>(user))
        reductions.push_back(reduction);
    }
  }
  llvm::SmallVector<mlir::Operation *> affineClosure;
  loop.walk(
      [&](mlir::Operation *operation) { affineClosure.push_back(operation); });
  mlir::GreedyRewriteConfig rewriteConfig;
  rewriteConfig.setScope(loop->getParentRegion())
      .setStrictness(mlir::GreedyRewriteStrictness::ExistingAndNewOps)
      .setRegionSimplificationLevel(mlir::GreedySimplifyRegionLevel::Disabled);
  mlir::RewritePatternSet affinePatterns(context);
  mlir::populateAffineToStdConversionPatterns(affinePatterns);
  mlir::FrozenRewritePatternSet frozenAffinePatterns(std::move(affinePatterns));
  if (mlir::failed(mlir::applyOpPatternsGreedily(
          affineClosure, frozenAffinePatterns, rewriteConfig)))
    return StructuredScopRefusalKind::VectorLoweringUnavailable;

  mlir::scf::ForOp lowered;
  mlir::Operation *operation =
      predecessor ? predecessor->getNextNode() : &block->front();
  while (operation && operation != successor) {
    if (auto candidate = llvm::dyn_cast<mlir::scf::ForOp>(operation)) {
      if (lowered)
        return invalid("Affine lowering produced an ambiguous SCF loop root");
      lowered = candidate;
    }
    operation = operation->getNextNode();
  }
  if (!lowered)
    return invalid("Affine lowering produced no SCF loop root");

  for (mlir::vector::ReductionOp reduction : reductions) {
    mlir::RewritePatternSet reductionPatterns(context);
    mlir::vector::populateBreakDownVectorReductionPatterns(
        reductionPatterns, static_cast<unsigned>(coordinate.shape.front()));
    mlir::FrozenRewritePatternSet frozenReductionPatterns(
        std::move(reductionPatterns));
    if (mlir::failed(mlir::applyOpPatternsGreedily(
            llvm::ArrayRef<mlir::Operation *>{reduction.getOperation()},
            frozenReductionPatterns, rewriteConfig)))
      return StructuredScopRefusalKind::VectorLoweringUnavailable;
  }
  for (mlir::Operation *closureOperation : vectorizedClosure(lowered)) {
    if (llvm::isa<mlir::vector::ReductionOp>(closureOperation))
      return StructuredScopRefusalKind::VectorLoweringUnavailable;
  }
  return lowered;
}

bool hasVectorType(mlir::Operation *operation) {
  for (mlir::Type type : operation->getOperandTypes())
    if (llvm::isa<mlir::VectorType>(type))
      return true;
  for (mlir::Type type : operation->getResultTypes())
    if (llvm::isa<mlir::VectorType>(type))
      return true;
  return false;
}

llvm::SmallVector<mlir::Operation *> vectorizedClosure(mlir::Operation *root) {
  llvm::SmallVector<mlir::Operation *> result;
  llvm::SmallVector<mlir::Operation *> pending;
  llvm::SmallPtrSet<mlir::Operation *, 32> seen;
  root->walk([&](mlir::Operation *operation) {
    if (seen.insert(operation).second) {
      result.push_back(operation);
      pending.push_back(operation);
    }
  });
  while (!pending.empty()) {
    mlir::Operation *operation = pending.pop_back_val();
    for (mlir::Value operand : operation->getOperands()) {
      mlir::Operation *definition = operand.getDefiningOp();
      if (definition && hasVectorType(definition) &&
          seen.insert(definition).second) {
        result.push_back(definition);
        pending.push_back(definition);
      }
    }
    for (mlir::Value value : operation->getResults()) {
      for (mlir::Operation *user : value.getUsers()) {
        if (seen.insert(user).second) {
          result.push_back(user);
          pending.push_back(user);
        }
      }
    }
  }
  return result;
}

llvm::Expected<std::uint64_t>
admittingVectorMemoryResources(mlir::vector::TransferReadOp read,
                               const FabricCapabilityIndex &fabric) {
  mlir::Block scratch;
  mlir::Location loc = read.getLoc();
  mlir::Value mem = scratch.addArgument(read.getBase().getType(), loc);
  mlir::Value address =
      scratch.addArgument(mlir::IndexType::get(read.getContext()), loc);
  mlir::Value ctrl =
      scratch.addArgument(mlir::NoneType::get(read.getContext()), loc);
  mlir::Value mask;
  if (read.getMask())
    mask = scratch.addArgument(read.getMask().getType(), loc);
  mlir::OpBuilder builder(read.getContext());
  builder.setInsertionPointToStart(&scratch);
  auto projected = dataflow::LoadOp::create(builder, loc, read.getVectorType(),
                                            builder.getNoneType(), mem, address,
                                            ctrl, mask, mlir::Attribute{});
  return fabric.admittingMemoryResourceCount(projected);
}

llvm::Expected<std::uint64_t>
admittingVectorMemoryResources(mlir::vector::TransferWriteOp write,
                               const FabricCapabilityIndex &fabric) {
  mlir::Block scratch;
  mlir::Location loc = write.getLoc();
  mlir::Value mem = scratch.addArgument(write.getBase().getType(), loc);
  mlir::Value address =
      scratch.addArgument(mlir::IndexType::get(write.getContext()), loc);
  mlir::Value data =
      scratch.addArgument(write.getValueToStore().getType(), loc);
  mlir::Value ctrl =
      scratch.addArgument(mlir::NoneType::get(write.getContext()), loc);
  mlir::Value mask;
  if (write.getMask())
    mask = scratch.addArgument(write.getMask().getType(), loc);
  mlir::OpBuilder builder(write.getContext());
  builder.setInsertionPointToStart(&scratch);
  auto projected =
      dataflow::StoreOp::create(builder, loc, builder.getNoneType(), mem,
                                address, data, ctrl, mask, mlir::Attribute{});
  return fabric.admittingMemoryResourceCount(projected);
}

llvm::Expected<std::uint64_t>
admittingStructuredActorResources(mlir::Operation *operation,
                                  const FabricCapabilityIndex &fabric) {
  const std::optional<dataflow::OperationSchemaId> schema =
      dataflow::operationSchemaOf(operation);
  if (!schema)
    return invalid("structured actor has no operation schema");
  if (dataflow::actorKind(*schema) ==
      dataflow::CanonicalDataflowActorKind::Memory)
    return fabric.admittingMemoryResourceCount(operation);
  auto projection = dataflow::projectRegisteredActorSchemaProjection(operation);
  if (!projection)
    return projection.takeError();
  if (*schema == dataflow::OperationSchemaId::ArithConstant) {
    projection->schema = dataflow::OperationSchemaId::DataflowConstant;
    projection->type = mlir::FunctionType::get(
        operation->getContext(), {mlir::NoneType::get(operation->getContext())},
        operation->getResultTypes());
  }
  auto indexBits = getIndexBitWidth(operation);
  if (!indexBits)
    return indexBits.takeError();
  return fabric.admittingOperationResourceCount(*projection, *indexBits);
}

llvm::Expected<bool>
fabricAdmitsVectorizedClosure(mlir::Operation *root,
                              const FabricCapabilityIndex &fabric) {
  bool admitted = true;
  llvm::Error queryError = llvm::Error::success();
  for (mlir::Operation *operation : vectorizedClosure(root)) {
    if (queryError || !admitted || !hasVectorType(operation))
      continue;
    if (auto read = llvm::dyn_cast<mlir::vector::TransferReadOp>(operation)) {
      auto resources = admittingVectorMemoryResources(read, fabric);
      if (!resources) {
        queryError = resources.takeError();
        break;
      }
      admitted = *resources != 0;
      continue;
    }
    if (auto write = llvm::dyn_cast<mlir::vector::TransferWriteOp>(operation)) {
      auto resources = admittingVectorMemoryResources(write, fabric);
      if (!resources) {
        queryError = resources.takeError();
        break;
      }
      admitted = *resources != 0;
      continue;
    }
    const std::optional<dataflow::OperationSchemaId> schema =
        dataflow::operationSchemaOf(operation);
    if (!schema) {
      if (!llvm::isa<mlir::scf::ForOp, mlir::scf::YieldOp>(operation))
        admitted = false;
      continue;
    }
    if (dataflow::actorKind(*schema) !=
        dataflow::CanonicalDataflowActorKind::Compute)
      continue;
    auto resources = admittingStructuredActorResources(operation, fabric);
    if (!resources) {
      queryError = resources.takeError();
      break;
    }
    admitted = *resources != 0;
  }
  if (queryError)
    return std::move(queryError);
  return admitted;
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
          StructuredScheduleDecisionKind::PolyhedralSchedule))
    return invalid("decision has an unknown kind");
  const bool factorless =
      decision.kind == StructuredScheduleDecisionKind::Interchange ||
      decision.kind == StructuredScheduleDecisionKind::Parallelize ||
      decision.kind == StructuredScheduleDecisionKind::ParallelizeNest ||
      decision.kind == StructuredScheduleDecisionKind::Vectorize ||
      decision.kind == StructuredScheduleDecisionKind::PolyhedralSchedule;
  if ((factorless && decision.factor != 0) ||
      (!factorless &&
       (decision.factor <= 1 ||
        decision.factor > maximumCanonicalStructuredScheduleFactor)))
    return invalid("decision has an invalid factor");
  if (decision.kind == StructuredScheduleDecisionKind::Vectorize) {
    if (!decision.vector)
      return invalid("vector decision has no vector coordinate");
    if (llvm::Error error = detail::validateStructuredVectorScheduleCoordinate(
            *decision.vector))
      return std::move(error);
  } else if (decision.vector) {
    return invalid("non-vector decision carries a vector coordinate");
  }
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
  if (decision.vector) {
    appendU32(static_cast<std::uint32_t>(decision.vector->shape.size()));
    for (std::uint64_t dimension : decision.vector->shape)
      appendU64(dimension);
    appendU32(static_cast<std::uint32_t>(decision.vector->tailPolicy));
    appendU64(decision.vector->requiredAlignmentBytes);
    appendU32(static_cast<std::uint32_t>(decision.vector->aliasPolicy));
    appendU32(static_cast<std::uint32_t>(decision.vector->reductionSchedule));
  }
  return bytes;
}

llvm::Expected<StructuredScheduleDecision>
adoptStructuredScheduleDecision(llvm::ArrayRef<std::uint8_t> canonicalBytes) {
  constexpr std::size_t scalarWireSize = structuredEntityRefWireSize + 12;
  if (canonicalBytes.size() < scalarWireSize)
    return invalid("decision payload is truncated");
  auto loop = decodeStructuredEntityRef(
      canonicalBytes.take_front(structuredEntityRefWireSize));
  if (!loop)
    return loop.takeError();
  if (loop->kind != StructuredEntityKind::Operation)
    return invalid("decision does not reference an operation");
  llvm::ArrayRef<std::uint8_t> suffix =
      canonicalBytes.drop_front(structuredEntityRefWireSize);
  std::size_t offset = 0;
  const auto readU32 = [&]() -> llvm::Expected<std::uint32_t> {
    if (suffix.size() - offset < 4)
      return invalid("decision payload has a truncated u32");
    std::uint32_t value = 0;
    for (std::uint8_t byte : suffix.slice(offset, 4))
      value = (value << 8) | byte;
    offset += 4;
    return value;
  };
  const auto readU64 = [&]() -> llvm::Expected<std::uint64_t> {
    if (suffix.size() - offset < 8)
      return invalid("decision payload has a truncated u64");
    std::uint64_t value = 0;
    for (std::uint8_t byte : suffix.slice(offset, 8))
      value = (value << 8) | byte;
    offset += 8;
    return value;
  };
  auto kind = readU32();
  if (!kind)
    return kind.takeError();
  if (*kind > static_cast<std::uint32_t>(
                  StructuredScheduleDecisionKind::PolyhedralSchedule))
    return invalid("decision payload has an unknown kind");
  auto factor = readU64();
  if (!factor)
    return factor.takeError();
  const auto typedKind = static_cast<StructuredScheduleDecisionKind>(*kind);
  const bool factorless =
      typedKind == StructuredScheduleDecisionKind::Interchange ||
      typedKind == StructuredScheduleDecisionKind::Parallelize ||
      typedKind == StructuredScheduleDecisionKind::ParallelizeNest ||
      typedKind == StructuredScheduleDecisionKind::Vectorize ||
      typedKind == StructuredScheduleDecisionKind::PolyhedralSchedule;
  if ((factorless && *factor != 0) || (!factorless && *factor <= 1))
    return invalid("decision payload has an invalid factor");
  std::optional<StructuredVectorScheduleCoordinate> vector;
  if (typedKind == StructuredScheduleDecisionKind::Vectorize) {
    auto rank = readU32();
    if (!rank)
      return rank.takeError();
    if (*rank != 1)
      return invalid("vector decision payload has an unsupported rank");
    std::vector<std::uint64_t> shape;
    shape.reserve(*rank);
    for (std::uint32_t dimension = 0; dimension != *rank; ++dimension) {
      auto size = readU64();
      if (!size)
        return size.takeError();
      shape.push_back(*size);
    }
    auto tail = readU32();
    if (!tail)
      return tail.takeError();
    auto alignment = readU64();
    if (!alignment)
      return alignment.takeError();
    auto alias = readU32();
    if (!alias)
      return alias.takeError();
    auto reduction = readU32();
    if (!reduction)
      return reduction.takeError();
    vector.emplace(StructuredVectorScheduleCoordinate{
        std::move(shape), static_cast<StructuredVectorTailPolicy>(*tail),
        *alignment, static_cast<StructuredVectorAliasPolicy>(*alias),
        static_cast<StructuredReductionSchedule>(*reduction)});
    if (llvm::Error error =
            detail::validateStructuredVectorScheduleCoordinate(*vector))
      return std::move(error);
  }
  if (offset != suffix.size())
    return invalid("decision payload has trailing bytes");
  StructuredScheduleDecision decision{*loop, typedKind, *factor,
                                      std::move(vector)};
  auto reencoded = encodeStructuredScheduleDecision(decision);
  if (!reencoded)
    return reencoded.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*reencoded) != canonicalBytes)
    return invalid("decision payload does not re-encode exactly");
  return decision;
}

llvm::Expected<StructuredScheduleDecisionDomain>
enumerateStructuredScheduleDecisions(
    const StructuredProgramCandidate &parent,
    const fabric::FinalizedFabricRoot &fabric,
    std::uint64_t scopeExpansionLimit,
    std::optional<StructuredEntityRef> schedulingScope) {
  if (scopeExpansionLimit == 0)
    return invalid("scope expansion limit must be positive");
  auto view = parent.view();
  if (!view)
    return view.takeError();
  mlir::Operation *scopeOperation = nullptr;
  if (schedulingScope) {
    if (schedulingScope->kind != StructuredEntityKind::Operation)
      return invalid("schedule scope does not reference an operation");
    auto resolved = view->resolve(*schedulingScope);
    if (!resolved)
      return resolved.takeError();
    scopeOperation = resolved->operation;
  }
  FabricCapabilityIndex capabilityIndex(fabric.view());
  std::vector<StructuredScheduleProposal> proposals;
  std::vector<StructuredPolyhedralScopView> polyhedralScops;
  std::vector<StructuredScopRefusal> refusals;
  std::uint64_t expanded = 0;
  std::uint64_t inspectedCoordinates = 0;
  std::uint64_t inspectedPolyhedralDependenceQueries = 0;
  const auto recordCoordinates = [&](std::uint64_t count) -> llvm::Error {
    const std::optional<std::uint64_t> next =
        llvm::checkedAddUnsigned(inspectedCoordinates, count);
    if (!next)
      return invalid("schedule-coordinate accounting overflows u64");
    inspectedCoordinates = *next;
    return llvm::Error::success();
  };
  const auto recordDependenceQueries = [&](std::uint64_t count) -> llvm::Error {
    const std::optional<std::uint64_t> next =
        llvm::checkedAddUnsigned(inspectedPolyhedralDependenceQueries, count);
    if (!next)
      return invalid("polyhedral dependence-query accounting overflows u64");
    inspectedPolyhedralDependenceQueries = *next;
    return llvm::Error::success();
  };
  for (const StructuredEntity &entity :
       view->entities(StructuredEntityKind::Operation)) {
    auto scfLoop = llvm::dyn_cast_or_null<mlir::scf::ForOp>(entity.operation);
    auto affineLoop =
        llvm::dyn_cast_or_null<mlir::affine::AffineForOp>(entity.operation);
    if (!scfLoop && !affineLoop)
      continue;
    if (scopeOperation && scopeOperation != entity.operation &&
        !scopeOperation->isAncestor(entity.operation))
      continue;
    if (expanded == scopeExpansionLimit)
      break;
    ++expanded;

    auto analysis = analyzeExactStructuredScop(parent, entity.reference);
    if (!analysis)
      return analysis.takeError();
    const auto appendGeneralScop = [&]() -> llvm::Expected<bool> {
      auto polyhedral =
          analyzeStructuredPolyhedralScop(parent, entity.reference);
      if (!polyhedral)
        return polyhedral.takeError();
      if (auto *general =
              std::get_if<StructuredPolyhedralScopView>(&*polyhedral)) {
        if (llvm::Error error =
                recordDependenceQueries(general->dependenceQueryCount))
          return std::move(error);
        auto frozen = std::make_shared<const StructuredPolyhedralScopView>(
            std::move(*general));
        polyhedralScops.push_back(*frozen);
        auto materializationRefusal = classifyPolyhedralScheduleMaterialization(
            entity.operation, *frozen);
        if (!materializationRefusal)
          return materializationRefusal.takeError();
        if (frozen->schedule.form !=
            StructuredPolyhedralScheduleForm::SourceOrder) {
          if (llvm::Error error = recordCoordinates(1))
            return std::move(error);
          if (!*materializationRefusal) {
            StructuredScheduleDecision decision{
                entity.reference,
                StructuredScheduleDecisionKind::PolyhedralSchedule, 0,
                std::nullopt};
            proposals.push_back(StructuredScheduleProposal(
                decision, nullptr, frozen, fabric.reference()));
          }
        }
        if (*materializationRefusal)
          refusals.push_back({entity.reference, **materializationRefusal});
        return true;
      }
      StructuredScopRefusal &polyhedralRefusal =
          std::get<StructuredScopRefusal>(*polyhedral);
      if (llvm::Error error =
              recordDependenceQueries(polyhedralRefusal.dependenceQueryCount))
        return std::move(error);
      refusals.push_back(std::move(polyhedralRefusal));
      return false;
    };
    if (auto *refusal = std::get_if<StructuredScopRefusal>(&*analysis)) {
      if (llvm::Error error =
              recordDependenceQueries(refusal->dependenceQueryCount))
        return std::move(error);
      const bool mayEnterGeneralDomain = [&] {
        switch (refusal->kind) {
        case StructuredScopRefusalKind::NonCanonicalIterationDomain:
        case StructuredScopRefusalKind::NestedControl:
        case StructuredScopRefusalKind::NonContiguousAccess:
        case StructuredScopRefusalKind::LoopCarriedMemoryDependence:
        case StructuredScopRefusalKind::AlignmentProofNotEstablished:
        case StructuredScopRefusalKind::NonUnitPhysicalStride:
        case StructuredScopRefusalKind::HeterogeneousElementWidth:
        case StructuredScopRefusalKind::NonLocalMemoryRoot:
        case StructuredScopRefusalKind::UnsupportedPhysicalOffset:
          return true;
        default:
          return false;
        }
      }();
      if (!mayEnterGeneralDomain) {
        refusals.push_back(*refusal);
      } else {
        auto admitted = appendGeneralScop();
        if (!admitted)
          return admitted.takeError();
      }
    } else {
      const ExactStructuredScopView &scop =
          std::get<ExactStructuredScopView>(*analysis);
      if (llvm::Error error =
              recordDependenceQueries(scop.dependenceQueryCount))
        return std::move(error);
      auto frozenScop = std::make_shared<const ExactStructuredScopView>(scop);
      if (!scop.constantTripCount) {
        auto admitted = appendGeneralScop();
        if (!admitted)
          return admitted.takeError();
      } else if (*scop.constantTripCount <= 1) {
        refusals.push_back(
            {entity.reference,
             StructuredScopRefusalKind::NonCanonicalIterationDomain});
      } else {
        const std::uint64_t maximumFactor = std::min(
            maximumCanonicalStructuredScheduleFactor, *scop.constantTripCount);
        if (llvm::Error error = recordCoordinates(maximumFactor - 1))
          return std::move(error);
        bool admitted = false;
        std::optional<StructuredScopRefusalKind> coordinateRefusal;
        for (std::uint64_t factor = 2; factor <= maximumFactor; ++factor) {
          auto coordinate = coordinateFor(scop, factor);
          if (!coordinate)
            return coordinate.takeError();
          if (auto *admittedCoordinate =
                  std::get_if<StructuredVectorScheduleCoordinate>(
                      &*coordinate)) {
            StructuredScheduleDecision decision{
                entity.reference, StructuredScheduleDecisionKind::Vectorize, 0,
                std::move(*admittedCoordinate)};
            proposals.push_back(StructuredScheduleProposal(
                decision, frozenScop, nullptr, fabric.reference()));
            admitted = true;
          } else {
            const StructuredScopRefusalKind refusal =
                std::get<StructuredScopRefusalKind>(*coordinate);
            if (!coordinateRefusal ||
                refusal == StructuredScopRefusalKind::UnsupportedTail)
              coordinateRefusal = refusal;
          }
        }
        if (!admitted)
          refusals.push_back({entity.reference,
                              coordinateRefusal.value_or(
                                  StructuredScopRefusalKind::UnsupportedTail)});
      }
    }
    if (affineLoop)
      continue;

    const auto appendScfProposal = [&](StructuredScheduleDecision decision) {
      proposals.push_back(StructuredScheduleProposal(decision, nullptr, nullptr,
                                                     fabric.reference()));
    };

    std::optional<std::uint64_t> tripCount = staticTripCount(scfLoop);
    if (tripCount && *tripCount > 1 && scfLoop.getInitArgs().empty()) {
      std::vector<std::uint64_t> factors = canonicalProperDivisors(*tripCount);
      if (llvm::Error error = recordCoordinates(factors.size() * 2))
        return std::move(error);
      for (std::uint64_t factor : factors)
        appendScfProposal({entity.reference,
                           StructuredScheduleDecisionKind::Tile, factor,
                           std::nullopt});

      auto capacity = aggregateUnrollCapacity(scfLoop, capabilityIndex);
      if (!capacity)
        return capacity.takeError();
      for (std::uint64_t factor : factors) {
        if (factor > *capacity)
          break;
        appendScfProposal({entity.reference,
                           StructuredScheduleDecisionKind::Unroll, factor,
                           std::nullopt});
      }
    }

    mlir::scf::ForOp inner;
    const bool perfectAdjacentNest = isPerfectAdjacentNest(scfLoop, inner);
    if (perfectAdjacentNest) {
      if (llvm::Error error = recordCoordinates(1))
        return std::move(error);
      const bool independentNest =
          raising::proveIndependentIterations(scfLoop) ==
              raising::ParallelDependenceResult::ProvenIndependent &&
          raising::proveIndependentIterations(inner) ==
              raising::ParallelDependenceResult::ProvenIndependent;
      if (independentNest)
        appendScfProposal({entity.reference,
                           StructuredScheduleDecisionKind::Interchange, 0,
                           std::nullopt});
      if (tripCount && *tripCount > 1) {
        std::vector<std::uint64_t> factors =
            canonicalProperDivisors(*tripCount);
        if (llvm::Error error = recordCoordinates(factors.size()))
          return std::move(error);
        if (independentNest && hasInvariantNestedLoopBounds(scfLoop)) {
          auto capacity = aggregateUnrollCapacity(scfLoop, capabilityIndex);
          if (!capacity)
            return capacity.takeError();
          for (std::uint64_t factor : factors) {
            if (factor > *capacity)
              break;
            appendScfProposal({entity.reference,
                               StructuredScheduleDecisionKind::UnrollAndJam,
                               factor, std::nullopt});
          }
        }
      }
    }
    if (scfLoop.getInitArgs().empty()) {
      if (llvm::Error error = recordCoordinates(1))
        return std::move(error);
      if (raising::proveIndependentIterations(scfLoop) ==
          raising::ParallelDependenceResult::ProvenIndependent)
        appendScfProposal({entity.reference,
                           StructuredScheduleDecisionKind::Parallelize, 0,
                           std::nullopt});
    }
    const llvm::SmallVector<mlir::scf::ForOp> rectangular =
        rectangularNest(scfLoop);
    if (rectangular.size() >= 2) {
      if (llvm::Error error = recordCoordinates(1))
        return std::move(error);
      if (raising::proveIndependentIterations(rectangular) ==
          raising::ParallelDependenceResult::ProvenIndependent)
        appendScfProposal({entity.reference,
                           StructuredScheduleDecisionKind::ParallelizeNest, 0,
                           std::nullopt});
    }
  }
  return StructuredScheduleDecisionDomain{
      std::move(proposals), std::move(polyhedralScops),
      std::move(refusals),  expanded,
      inspectedCoordinates, inspectedPolyhedralDependenceQueries};
}

namespace {

llvm::Expected<MaterializedStructuredScheduleCandidate>
materializeStructuredScheduleImpl(
    const StructuredProgramCandidate &parent,
    const StructuredScheduleDecision &decision,
    const ExactStructuredScopView *frozenScop,
    const StructuredPolyhedralScopView *frozenPolyhedralScop,
    const FabricCapabilityIndex *fabric,
    std::optional<StructuredEntityRef> trackedSpatialRegion,
    llvm::ArrayRef<StructuredOperationSourceProvenance> sourceProvenance) {
  auto encoded = encodeStructuredScheduleDecision(decision);
  if (!encoded)
    return encoded.takeError();
  std::optional<ExactStructuredScopView> vectorSource;
  const ExactStructuredScopView *exactVectorSource = frozenScop;
  if (decision.kind == StructuredScheduleDecisionKind::Vectorize) {
    if (!exactVectorSource) {
      auto analysis = analyzeExactStructuredScop(parent, decision.loop);
      if (!analysis)
        return analysis.takeError();
      if (auto *refusal = std::get_if<StructuredScopRefusal>(&*analysis))
        return invalid("selected vector SCoP is locally refused with kind " +
                       llvm::Twine(static_cast<std::uint32_t>(refusal->kind)));
      vectorSource.emplace(std::get<ExactStructuredScopView>(*analysis));
      exactVectorSource = &*vectorSource;
    }
    if (exactVectorSource->loop != decision.loop)
      return invalid("frozen vector SCoP belongs to another source loop");
    auto expected =
        coordinateFor(*exactVectorSource, decision.vector->shape.front());
    if (!expected)
      return expected.takeError();
    const auto *canonical =
        std::get_if<StructuredVectorScheduleCoordinate>(&*expected);
    if (!canonical || !(*canonical == *decision.vector))
      return invalid("vector coordinate is not canonical for its source SCoP");
  }
  std::optional<StructuredPolyhedralScopView> polyhedralSource;
  const StructuredPolyhedralScopView *exactPolyhedralSource =
      frozenPolyhedralScop;
  if (decision.kind == StructuredScheduleDecisionKind::PolyhedralSchedule) {
    if (!exactPolyhedralSource) {
      auto analysis = analyzeStructuredPolyhedralScop(parent, decision.loop);
      if (!analysis)
        return analysis.takeError();
      if (auto *refusal = std::get_if<StructuredScopRefusal>(&*analysis))
        return invalid(
            "selected polyhedral SCoP is locally refused with kind " +
            llvm::Twine(static_cast<std::uint32_t>(refusal->kind)));
      polyhedralSource.emplace(
          std::get<StructuredPolyhedralScopView>(std::move(*analysis)));
      exactPolyhedralSource = &*polyhedralSource;
    }
    if (exactPolyhedralSource->root != decision.loop)
      return invalid("frozen polyhedral SCoP belongs to another source loop");
    if (exactPolyhedralSource->schedule.form ==
        StructuredPolyhedralScheduleForm::SourceOrder)
      return invalid("polyhedral decision has no transform schedule form");
  }

  mlir::Operation *selectedLoop = nullptr;
  mlir::Operation *clonedSpatialRegion = nullptr;
  mlir::IRMapping cloneMapping;
  auto clone = cloneAndResolveLoop(parent, decision.loop, trackedSpatialRegion,
                                   sourceProvenance, cloneMapping, selectedLoop,
                                   clonedSpatialRegion);
  if (!clone)
    return clone.takeError();
  auto scfLoop = llvm::dyn_cast_or_null<mlir::scf::ForOp>(selectedLoop);
  const bool insideSpatialRegion =
      static_cast<bool>(selectedLoop->getParentOfType<loom::SpatialRegionOp>());

  switch (decision.kind) {
  case StructuredScheduleDecisionKind::Tile:
    if (!scfLoop)
      return invalid("tile decision does not reference scf.for");
    if (llvm::Error error = applyTile(scfLoop, decision.factor))
      return std::move(error);
    break;
  case StructuredScheduleDecisionKind::Unroll:
    if (!scfLoop)
      return invalid("unroll decision does not reference scf.for");
    if (llvm::Error error = applyUnroll(scfLoop, decision.factor))
      return std::move(error);
    break;
  case StructuredScheduleDecisionKind::Interchange:
    if (!scfLoop)
      return invalid("interchange decision does not reference scf.for");
    if (decision.factor != 0)
      return invalid("interchange decision carries a factor");
    if (llvm::Error error = applyInterchange(scfLoop))
      return std::move(error);
    break;
  case StructuredScheduleDecisionKind::UnrollAndJam:
    if (!scfLoop)
      return invalid("unroll-and-jam decision does not reference scf.for");
    if (llvm::Error error = applyUnrollAndJam(scfLoop, decision.factor))
      return std::move(error);
    break;
  case StructuredScheduleDecisionKind::Parallelize:
    if (!scfLoop)
      return invalid("parallelize decision does not reference scf.for");
    if (decision.factor != 0)
      return invalid("parallelize decision carries a factor");
    {
      mlir::Block *parentBlock = scfLoop->getBlock();
      mlir::Operation *successor = scfLoop->getNextNode();
      if (mlir::failed(raising::materializeIndependentLoopAsForall(scfLoop)))
        return invalid("SCF parallelization rejected the selected decision");
      mlir::Operation *replacement =
          successor ? successor->getPrevNode() : &parentBlock->back();
      auto parallel = llvm::dyn_cast_or_null<mlir::scf::ForallOp>(replacement);
      if (!parallel)
        return invalid("SCF parallelization did not produce one forall");
      if (insideSpatialRegion)
        if (llvm::Error error =
                materializeOwnedSpatialForallThreadDomain(parallel))
          return std::move(error);
    }
    break;
  case StructuredScheduleDecisionKind::ParallelizeNest:
    if (!scfLoop)
      return invalid("parallel-nest decision does not reference scf.for");
    if (decision.factor != 0)
      return invalid("parallel-nest decision carries a factor");
    if (auto parallel = applyParallelizeNest(scfLoop)) {
      if (insideSpatialRegion)
        if (llvm::Error error =
                materializeOwnedSpatialForallThreadDomain(*parallel))
          return std::move(error);
    } else {
      return parallel.takeError();
    }
    break;
  case StructuredScheduleDecisionKind::Vectorize:
    if (!exactVectorSource || !decision.vector)
      return invalid("vector decision has no exact SCoP coordinate");
    {
      auto projected = projectExactStructuredScopToAffine(selectedLoop);
      if (!projected)
        return projected.takeError();
      auto vectorized = applyVectorize(*projected, *decision.vector);
      if (!vectorized)
        return vectorized.takeError();
      if (const auto *refusal =
              std::get_if<StructuredScopRefusalKind>(&*vectorized)) {
        if (fabric)
          return llvm::make_error<StructuredScheduleProposalRefusal>(
              decision.loop, *refusal);
        return invalid("Affine provider rejected the selected coordinate");
      }
      mlir::affine::AffineForOp vectorizedLoop =
          std::get<mlir::affine::AffineForOp>(*vectorized);
      if (decision.vector->tailPolicy ==
          StructuredVectorTailPolicy::ReductionMask)
        if (llvm::Error error = attachTailMask(vectorizedLoop))
          return std::move(error);
      if (llvm::Error error = verifyStructuredVectorScheduleMaterialization(
              *exactVectorSource, *decision.vector, clone->get()))
        return std::move(error);
      auto lowered = lowerVectorizedScop(vectorizedLoop, *decision.vector);
      if (!lowered)
        return lowered.takeError();
      if (const auto *refusal =
              std::get_if<StructuredScopRefusalKind>(&*lowered)) {
        if (fabric)
          return llvm::make_error<StructuredScheduleProposalRefusal>(
              decision.loop, *refusal);
        return invalid("vector lowering rejected the selected coordinate");
      }
      mlir::scf::ForOp loweredLoop = std::get<mlir::scf::ForOp>(*lowered);
      if (fabric && decision.vector->tailPolicy ==
                        StructuredVectorTailPolicy::ReductionMask)
        return llvm::make_error<StructuredScheduleProposalRefusal>(
            decision.loop,
            StructuredScopRefusalKind::VectorLoweringUnavailable);
      if (fabric) {
        auto admitted =
            fabricAdmitsVectorizedClosure(loweredLoop.getOperation(), *fabric);
        if (!admitted)
          return admitted.takeError();
        if (!*admitted)
          return llvm::make_error<StructuredScheduleProposalRefusal>(
              decision.loop,
              StructuredScopRefusalKind::FabricCapabilityUnavailable);
      }
    }
    if (llvm::Error error = verifyStructuredVectorScheduleMaterialization(
            *exactVectorSource, *decision.vector, clone->get()))
      return std::move(error);
    break;
  case StructuredScheduleDecisionKind::PolyhedralSchedule:
    if (!exactPolyhedralSource)
      return invalid("polyhedral decision has no exact SCoP schedule");
    if (decision.factor != 0 || decision.vector)
      return invalid("polyhedral decision carries a scalar coordinate");
    if (canMaterializeCanonicalPolyhedralSchedule(selectedLoop,
                                                  *exactPolyhedralSource)) {
      if (llvm::Error error =
              applyPolyhedralSchedule(selectedLoop, *exactPolyhedralSource))
        return std::move(error);
    } else {
      auto parentView = parent.view();
      if (!parentView)
        return parentView.takeError();
      llvm::SmallVector<mlir::Operation *> materializedOperations;
      auto materialized = detail::materializePinnedIslSchedule(
          selectedLoop, *exactPolyhedralSource, *parentView, cloneMapping,
          materializedOperations);
      if (!materialized)
        return materialized.takeError();
      if (*materialized) {
        if (fabric)
          return llvm::make_error<StructuredScheduleProposalRefusal>(
              decision.loop, **materialized);
        return invalid("polyhedral schedule materialization was refused");
      }
      if (fabric) {
        for (mlir::Operation *operation : materializedOperations) {
          if (!dataflow::operationSchemaOf(operation))
            continue;
          auto resourceCount =
              admittingStructuredActorResources(operation, *fabric);
          if (!resourceCount)
            return resourceCount.takeError();
          if (*resourceCount == 0)
            return llvm::make_error<StructuredScheduleProposalRefusal>(
                decision.loop,
                StructuredScopRefusalKind::FabricCapabilityUnavailable);
        }
      }
    }
    break;
  }
  std::optional<std::string> parallelRejection;
  clone->get().walk([&](loom::SpatialRegionOp spatial) {
    if (!parallelRejection)
      parallelRejection =
          lowering::explainSpatialCarrierParallelRejection(spatial);
  });
  if (parallelRejection)
    return llvm::make_error<SpatialOwnershipCandidateRejection>(
        SpatialOwnershipCandidateRejectionKind::NonFinalizable,
        std::move(*parallelRejection));
  if (mlir::failed(mlir::verify(**clone)))
    return invalid("materialized schedule candidate does not verify");
  llvm::SmallVector<mlir::Operation *, 1> trackedOperations;
  if (clonedSpatialRegion)
    trackedOperations.push_back(clonedSpatialRegion);
  auto finalized = finalizeStructuredProgramWithTrackedEntities(
      clone->get(), {}, trackedOperations);
  if (!finalized)
    return finalized.takeError();
  if (finalized->trackedOperations.size() != trackedOperations.size())
    return invalid("tracked schedule projection changed cardinality");
  std::optional<StructuredEntityRef> projectedSpatial;
  if (clonedSpatialRegion)
    projectedSpatial = finalized->trackedOperations.front();
  return MaterializedStructuredScheduleCandidate{
      std::move(finalized->artifact), std::move(projectedSpatial),
      std::move(finalized->sourceProvenance)};
}

} // namespace

llvm::Expected<MaterializedStructuredScheduleCandidate>
materializeStructuredScheduleDecision(
    const StructuredProgramCandidate &parent,
    const StructuredScheduleDecision &decision,
    std::optional<StructuredEntityRef> trackedSpatialRegion,
    llvm::ArrayRef<StructuredOperationSourceProvenance> sourceProvenance) {
  return materializeStructuredScheduleImpl(parent, decision, nullptr, nullptr,
                                           nullptr, trackedSpatialRegion,
                                           sourceProvenance);
}

llvm::Expected<MaterializedStructuredScheduleCandidate>
materializeStructuredScheduleProposal(
    const StructuredProgramCandidate &parent,
    const StructuredScheduleProposal &proposal,
    const fabric::FinalizedFabricRoot &fabric,
    std::optional<StructuredEntityRef> trackedSpatialRegion,
    llvm::ArrayRef<StructuredOperationSourceProvenance> sourceProvenance) {
  if (proposal.decision_.loop.parent != parent.identity())
    return invalid("schedule proposal belongs to another parent");
  if (proposal.fabric_ != fabric.reference())
    return invalid("schedule proposal belongs to another Fabric");
  if ((proposal.decision_.kind == StructuredScheduleDecisionKind::Vectorize) !=
      static_cast<bool>(proposal.exactScop_))
    return invalid("schedule proposal has an inconsistent frozen SCoP");
  if ((proposal.decision_.kind ==
       StructuredScheduleDecisionKind::PolyhedralSchedule) !=
      static_cast<bool>(proposal.polyhedralScop_))
    return invalid("schedule proposal has an inconsistent polyhedral SCoP");
  if (proposal.exactScop_ && proposal.polyhedralScop_)
    return invalid("schedule proposal has competing exact SCoP views");
  FabricCapabilityIndex capabilityIndex(fabric.view());
  return materializeStructuredScheduleImpl(
      parent, proposal.decision_, proposal.exactScop_.get(),
      proposal.polyhedralScop_.get(), &capabilityIndex, trackedSpatialRegion,
      sourceProvenance);
}

llvm::Error
verifyStructuredScheduleDerivation(const StructuredProgramCandidate &parent,
                                   const fabric::FinalizedFabricRoot &fabric,
                                   const StructuredScheduleDecision &decision,
                                   const StructuredProgramCandidate &child) {
  if (decision.loop.parent != parent.identity())
    return invalid("schedule derivation decision belongs to another parent");
  if (parent.identity() == child.identity())
    return invalid("schedule derivation is a self edge");
  auto domain = enumerateStructuredScheduleDecisions(
      parent, fabric, std::numeric_limits<std::uint64_t>::max(), decision.loop);
  if (!domain)
    return domain.takeError();
  auto admitted = llvm::find_if(
      domain->proposals, [&](const StructuredScheduleProposal &proposal) {
        return proposal.decision() == decision;
      });
  if (admitted == domain->proposals.end())
    return invalid("schedule derivation decision is outside the admitted "
                   "domain: loop=" +
                   llvm::Twine(decision.loop.ordinal) + " kind=" +
                   llvm::Twine(static_cast<std::uint32_t>(decision.kind)) +
                   " factor=" + llvm::Twine(decision.factor));
  auto replayed =
      materializeStructuredScheduleProposal(parent, *admitted, fabric);
  if (!replayed)
    return replayed.takeError();
  if (replayed->structuredProgram.identity() != child.identity() ||
      replayed->structuredProgram.canonicalBytes().bytes() !=
          child.canonicalBytes().bytes())
    return invalid("schedule derivation does not replay to its exact child");
  return llvm::Error::success();
}

} // namespace loom::frontend
