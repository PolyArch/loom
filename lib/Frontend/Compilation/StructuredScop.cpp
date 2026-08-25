#include "Frontend/Compilation/StructuredScop.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/OperationSchema.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Conversion/SCFToAffine/SCFToAffine.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom::frontend {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "structured_scop_invalid: " + message);
}

StructuredScopAnalysisOutcome refuse(const StructuredEntityRef &loop,
                                     StructuredScopRefusalKind kind) {
  return StructuredScopRefusal{loop, kind};
}

bool hasReassociation(mlir::Operation *operation) {
  auto fastMath =
      llvm::dyn_cast<mlir::arith::ArithFastMathInterface>(operation);
  if (!fastMath)
    return false;
  mlir::arith::FastMathFlagsAttr attribute = fastMath.getFastMathFlagsAttr();
  return attribute &&
         mlir::arith::bitEnumContainsAny(attribute.getValue(),
                                         mlir::arith::FastMathFlags::reassoc);
}

bool hasIntegerOverflowContract(mlir::Operation *operation) {
  if (auto add = llvm::dyn_cast<mlir::arith::AddIOp>(operation))
    return add.getOverflowFlags() != mlir::arith::IntegerOverflowFlags::none;
  if (auto multiply = llvm::dyn_cast<mlir::arith::MulIOp>(operation))
    return multiply.getOverflowFlags() !=
           mlir::arith::IntegerOverflowFlags::none;
  return false;
}

std::optional<std::uint64_t> knownAlignment(mlir::Value value) {
  for (unsigned depth = 0; depth != 16; ++depth) {
    if (auto alignment = value.getDefiningOp<mlir::memref::AssumeAlignmentOp>())
      return alignment.getAlignment();
    auto result = llvm::dyn_cast<mlir::OpResult>(value);
    if (!result)
      return std::nullopt;
    auto distinct = llvm::dyn_cast_or_null<mlir::memref::DistinctObjectsOp>(
        result.getOwner());
    if (!distinct || result.getResultNumber() >= distinct.getNumOperands())
      return std::nullopt;
    value = distinct.getOperand(result.getResultNumber());
  }
  return std::nullopt;
}

std::optional<std::uint64_t> localBoundaryArgument(mlir::Value value,
                                                   mlir::Operation *loop) {
  for (unsigned depth = 0; depth != 16; ++depth) {
    if (auto blockArgument = llvm::dyn_cast<mlir::BlockArgument>(value)) {
      if (blockArgument.getOwner() != loop->getBlock())
        return std::nullopt;
      return blockArgument.getArgNumber();
    }
    auto result = llvm::dyn_cast<mlir::OpResult>(value);
    if (!result)
      return std::nullopt;
    if (llvm::isa<mlir::memref::AssumeAlignmentOp>(result.getOwner())) {
      value = result.getOwner()->getOperand(0);
      continue;
    }
    auto distinct =
        llvm::dyn_cast<mlir::memref::DistinctObjectsOp>(result.getOwner());
    if (!distinct || result.getResultNumber() >= distinct.getNumOperands())
      return std::nullopt;
    value = distinct.getOperand(result.getResultNumber());
  }
  return std::nullopt;
}

llvm::Expected<std::uint64_t> elementBytes(mlir::Type type,
                                           mlir::Operation *anchor) {
  std::uint64_t bits = 0;
  if (llvm::isa<mlir::IndexType>(type)) {
    auto width = getIndexBitWidth(anchor);
    if (!width)
      return width.takeError();
    bits = *width;
  } else if (type.isIntOrFloat()) {
    bits = type.getIntOrFloatBitWidth();
  } else {
    return invalid("affine access element is not scalar integer or floating");
  }
  if (bits == 0 || bits % 8 != 0)
    return invalid("affine access element has no whole-byte representation");
  return bits / 8;
}

std::optional<StructuredScopRefusalKind>
physicalLayoutRefusal(mlir::MemRefType type) {
  llvm::SmallVector<std::int64_t> strides;
  std::int64_t offset = 0;
  if (type.getRank() != 1 ||
      mlir::failed(type.getStridesAndOffset(strides, offset)) ||
      strides.size() != 1 || strides.front() != 1)
    return StructuredScopRefusalKind::NonUnitPhysicalStride;
  if (offset != 0)
    return StructuredScopRefusalKind::UnsupportedPhysicalOffset;
  return std::nullopt;
}

template <typename AccessOp>
bool hasCanonicalContiguousAccess(AccessOp access,
                                  mlir::affine::AffineForOp loop) {
  const mlir::AffineMap map = access.getAffineMap();
  return map.getNumDims() == 1 && map.getNumSymbols() == 0 &&
         map.getNumResults() == 1 &&
         map.getResult(0) == mlir::getAffineDimExpr(0, loop.getContext()) &&
         access.getMapOperands().size() == 1 &&
         access.getMapOperands().front() == loop.getInductionVar();
}

struct SourceAccess final {
  mlir::Value memref;
  bool writes = false;
  std::uint64_t statementOrdinal = 0;
  std::optional<std::uint64_t> storedStatementOrdinal;
};

std::optional<SourceAccess> sourceAccess(
    mlir::Operation *operation, std::uint64_t statementOrdinal,
    const llvm::DenseMap<mlir::Operation *, std::uint64_t> &statementOrdinals) {
  if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(operation))
    return SourceAccess{load.getMemref(), false, statementOrdinal,
                        std::nullopt};
  if (auto store = llvm::dyn_cast<mlir::memref::StoreOp>(operation)) {
    auto stored = llvm::dyn_cast<mlir::OpResult>(store.getValue());
    auto owner = stored ? statementOrdinals.find(stored.getOwner())
                        : statementOrdinals.end();
    if (owner == statementOrdinals.end())
      return std::nullopt;
    return SourceAccess{store.getMemref(), true, statementOrdinal,
                        owner->second};
  }
  if (auto load = llvm::dyn_cast<mlir::affine::AffineLoadOp>(operation))
    return SourceAccess{load.getMemRef(), false, statementOrdinal,
                        std::nullopt};
  if (auto store = llvm::dyn_cast<mlir::affine::AffineStoreOp>(operation)) {
    auto stored = llvm::dyn_cast<mlir::OpResult>(store.getValueToStore());
    auto owner = stored ? statementOrdinals.find(stored.getOwner())
                        : statementOrdinals.end();
    if (owner == statementOrdinals.end())
      return std::nullopt;
    return SourceAccess{store.getMemRef(), true, statementOrdinal,
                        owner->second};
  }
  return std::nullopt;
}

struct AccessProof final {
  mlir::affine::MemRefAccess access;
  mlir::Value memref;
  bool writes = false;
  StructuredScopAccessView view;
};

llvm::Expected<std::variant<AccessProof, StructuredScopRefusalKind>>
analyzeAccess(mlir::Operation *operation, mlir::affine::AffineForOp loop,
              const SourceAccess &source, mlir::Operation *sourceLoop) {
  mlir::Value memref;
  bool writes = false;
  if (auto load = llvm::dyn_cast<mlir::affine::AffineLoadOp>(operation)) {
    if (auto refusal = physicalLayoutRefusal(load.getMemRefType()))
      return *refusal;
    if (!hasCanonicalContiguousAccess(load, loop))
      return StructuredScopRefusalKind::NonContiguousAccess;
    memref = load.getMemRef();
  } else if (auto store =
                 llvm::dyn_cast<mlir::affine::AffineStoreOp>(operation)) {
    if (auto refusal = physicalLayoutRefusal(store.getMemRefType()))
      return *refusal;
    if (!hasCanonicalContiguousAccess(store, loop))
      return StructuredScopRefusalKind::NonContiguousAccess;
    memref = store.getMemRef();
    writes = true;
  } else {
    return StructuredScopRefusalKind::UnsupportedEffect;
  }
  if (writes != source.writes)
    return invalid("SCF-to-Affine projection changed access direction");

  mlir::affine::MemRefAccess access(operation);
  mlir::presburger::IntegerRelation relation(
      mlir::presburger::PresburgerSpace::getRelationSpace());
  if (mlir::failed(access.getAccessRelation(relation)))
    return StructuredScopRefusalKind::AccessRelationProofNotEstablished;
  auto bytes = elementBytes(
      llvm::cast<mlir::MemRefType>(memref.getType()).getElementType(),
      operation);
  if (!bytes)
    return bytes.takeError();
  const std::optional<std::uint64_t> alignment = knownAlignment(source.memref);
  if (!alignment)
    return StructuredScopRefusalKind::AlignmentProofNotEstablished;
  const std::optional<std::uint64_t> boundary =
      localBoundaryArgument(source.memref, sourceLoop);
  if (!boundary)
    return StructuredScopRefusalKind::NonLocalMemoryRoot;

  StructuredScopAccessView view;
  view.kind =
      writes ? StructuredScopAccessKind::Write : StructuredScopAccessKind::Read;
  view.statementOrdinal = source.statementOrdinal;
  view.relationDimensionCount = relation.getNumDimVars();
  view.relationSymbolCount = relation.getNumSymbolVars();
  view.relationConstraintCount = relation.getNumConstraints();
  view.elementBytes = *bytes;
  view.alignmentBytes = *alignment;
  view.memoryBoundaryArgument = *boundary;
  view.storedStatementOrdinal = source.storedStatementOrdinal;
  return AccessProof{access, memref, writes, std::move(view)};
}

llvm::Expected<
    std::variant<StructuredReductionSchedule, StructuredScopRefusalKind>>
analyzeReductions(mlir::affine::AffineForOp loop,
                  llvm::ArrayRef<mlir::affine::LoopReduction> reductions) {
  if (loop.getNumIterOperands() != reductions.size())
    return StructuredScopRefusalKind::UnsupportedReduction;
  if (reductions.empty())
    return StructuredReductionSchedule::None;
  if (reductions.size() != 1)
    return StructuredScopRefusalKind::UnsupportedReduction;

  const mlir::affine::LoopReduction &reduction = reductions.front();
  if (reduction.iterArgPosition >= loop.getInits().size())
    return StructuredScopRefusalKind::UnsupportedReduction;
  mlir::Value init = loop.getInits()[reduction.iterArgPosition];
  auto constant = init.getDefiningOp<mlir::arith::ConstantOp>();
  mlir::OpBuilder builder(loop);
  const mlir::TypedAttr identity = mlir::arith::getIdentityValueAttr(
      reduction.kind, init.getType(), builder, init.getLoc());
  if (!constant || constant.getValue() != identity)
    return StructuredScopRefusalKind::UnsupportedReduction;

  bool sawInteger = false;
  bool sawFloating = false;
  for (const mlir::affine::LoopReduction &reduction : reductions) {
    llvm::SmallVector<mlir::Operation *> combiners;
    mlir::Value reduced = mlir::matchReduction(
        loop.getRegionIterArgs(), reduction.iterArgPosition, combiners);
    if (!reduced || combiners.size() != 1)
      return StructuredScopRefusalKind::UnsupportedReduction;
    mlir::Type type =
        loop.getRegionIterArgs()[reduction.iterArgPosition].getType();
    if (llvm::isa<mlir::IntegerType, mlir::IndexType>(type)) {
      if (hasIntegerOverflowContract(combiners.front()))
        return StructuredScopRefusalKind::IntegerOverflowReduction;
      sawInteger = true;
      continue;
    }
    if (!llvm::isa<mlir::FloatType>(type))
      return StructuredScopRefusalKind::UnsupportedReduction;
    sawFloating = true;
    if (!hasReassociation(combiners.front()))
      return StructuredScopRefusalKind::StrictFloatingReduction;
  }
  if (sawInteger && sawFloating)
    return StructuredScopRefusalKind::UnsupportedReduction;
  return sawFloating ? StructuredReductionSchedule::FloatingReassociated
                     : StructuredReductionSchedule::IntegerAssociative;
}

llvm::Expected<mlir::affine::AffineForOp>
findProjectedLoop(mlir::Block *block, mlir::Operation *predecessor,
                  mlir::Operation *successor) {
  mlir::affine::AffineForOp result;
  mlir::Operation *operation =
      predecessor ? predecessor->getNextNode() : &block->front();
  while (operation && operation != successor) {
    if (auto loop = llvm::dyn_cast<mlir::affine::AffineForOp>(operation)) {
      if (result)
        return invalid("SCF provider produced an ambiguous affine loop root");
      result = loop;
    }
    operation = operation->getNextNode();
  }
  if (!result)
    return invalid("SCF provider produced no affine loop root");
  return result;
}

llvm::Expected<StructuredScopAnalysisOutcome> analyzeProjectedScop(
    const StructuredEntityRef &loopReference, mlir::Operation *sourceLoop,
    mlir::affine::AffineForOp loop, mlir::ModuleOp providerModule) {
  if (loop->getParentOfType<mlir::affine::AffineForOp>() ||
      loop->getParentOfType<mlir::affine::AffineIfOp>())
    return refuse(loopReference, StructuredScopRefusalKind::NestedAffineRoot);
  if (loop.getStepAsInt() != 1 || !loop.hasConstantLowerBound() ||
      loop.getConstantLowerBound() != 0)
    return refuse(loopReference,
                  StructuredScopRefusalKind::NonCanonicalIterationDomain);

  llvm::SmallVector<mlir::Operation *> domainOperations = {loop.getOperation()};
  mlir::affine::FlatAffineValueConstraints domain;
  if (mlir::failed(mlir::affine::getIndexSet(domainOperations, &domain)))
    return refuse(loopReference,
                  StructuredScopRefusalKind::DomainProofNotEstablished);

  std::vector<mlir::Operation *> sourceStatementOps;
  llvm::DenseMap<mlir::Operation *, std::uint64_t> sourceStatementOrdinals;
  mlir::Region &sourceRegion =
      llvm::isa<mlir::scf::ForOp>(sourceLoop)
          ? llvm::cast<mlir::scf::ForOp>(sourceLoop).getRegion()
          : llvm::cast<mlir::affine::AffineForOp>(sourceLoop).getRegion();
  for (mlir::Operation &operation : sourceRegion.front().without_terminator()) {
    if (operation.getNumRegions() != 0)
      return refuse(loopReference, StructuredScopRefusalKind::NestedControl);
    sourceStatementOrdinals.try_emplace(&operation, sourceStatementOps.size());
    sourceStatementOps.push_back(&operation);
  }
  std::vector<std::optional<SourceAccess>> sourceAccesses;
  sourceAccesses.reserve(sourceStatementOps.size());
  for (auto [ordinal, operation] : llvm::enumerate(sourceStatementOps))
    sourceAccesses.push_back(
        sourceAccess(operation, ordinal, sourceStatementOrdinals));

  std::vector<AccessProof> accesses;
  std::vector<StructuredScopComputeView> computes;
  std::uint64_t projectedStatements = 0;
  for (mlir::Operation &operation : loop.getBody()->without_terminator()) {
    if (operation.getNumRegions() != 0)
      return refuse(loopReference, StructuredScopRefusalKind::NestedControl);
    if (projectedStatements >= sourceStatementOps.size())
      return invalid("SCF-to-Affine projection added a statement");
    mlir::Operation *sourceOperation = sourceStatementOps[projectedStatements];
    const std::optional<SourceAccess> &source =
        sourceAccesses[projectedStatements];
    if (llvm::isa<mlir::affine::AffineLoadOp, mlir::affine::AffineStoreOp>(
            operation)) {
      if (!source)
        return refuse(loopReference,
                      StructuredScopRefusalKind::UnsupportedOperation);
      auto access = analyzeAccess(&operation, loop, *source, sourceLoop);
      if (!access)
        return access.takeError();
      if (auto *refusal = std::get_if<StructuredScopRefusalKind>(&*access))
        return refuse(loopReference, *refusal);
      accesses.push_back(std::move(std::get<AccessProof>(*access)));
      ++projectedStatements;
      continue;
    }
    if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(operation)) {
      if (auto refusal = physicalLayoutRefusal(load.getMemRefType()))
        return refuse(loopReference, *refusal);
      return refuse(loopReference,
                    StructuredScopRefusalKind::NonContiguousAccess);
    }
    if (auto store = llvm::dyn_cast<mlir::memref::StoreOp>(operation)) {
      if (auto refusal = physicalLayoutRefusal(store.getMemRefType()))
        return refuse(loopReference, *refusal);
      return refuse(loopReference,
                    StructuredScopRefusalKind::NonContiguousAccess);
    }
    if (source)
      return invalid("SCF-to-Affine projection removed a memory access");
    if (!mlir::isMemoryEffectFree(&operation))
      return refuse(loopReference,
                    StructuredScopRefusalKind::UnsupportedEffect);
    if (!dataflow::isCanonicalDataflowActor(sourceOperation) ||
        !dataflow::isCanonicalDataflowActor(&operation))
      return refuse(loopReference,
                    StructuredScopRefusalKind::UnsupportedOperation);
    auto sourceProjection =
        dataflow::projectRegisteredActorSchemaProjection(sourceOperation);
    if (!sourceProjection)
      return sourceProjection.takeError();
    auto providerProjection =
        dataflow::projectRegisteredActorSchemaProjection(&operation);
    if (!providerProjection)
      return providerProjection.takeError();
    if (sourceProjection->schema != providerProjection->schema ||
        !(sourceProjection->payload == providerProjection->payload))
      return invalid("SCF-to-Affine projection changed actor semantics");
    StructuredScopComputeView compute;
    compute.statementOrdinal = projectedStatements;
    compute.schema = sourceProjection->schema;
    compute.payload = sourceProjection->payload;
    for (mlir::Value operand : sourceOperation->getOperands()) {
      if (auto result = llvm::dyn_cast<mlir::OpResult>(operand)) {
        auto statement = sourceStatementOrdinals.find(result.getOwner());
        if (statement == sourceStatementOrdinals.end() ||
            statement->second >= projectedStatements)
          return refuse(loopReference,
                        StructuredScopRefusalKind::UnsupportedOperation);
        compute.operandStatements.push_back(statement->second);
        continue;
      }
      auto argument = llvm::dyn_cast<mlir::BlockArgument>(operand);
      if (!argument || argument.getOwner() != &sourceRegion.front() ||
          argument.getArgNumber() == 0)
        return refuse(loopReference,
                      StructuredScopRefusalKind::UnsupportedOperation);
      compute.operandStatements.push_back(std::nullopt);
    }
    computes.push_back(std::move(compute));
    ++projectedStatements;
  }
  if (projectedStatements != sourceStatementOps.size())
    return invalid("SCF-to-Affine projection changed statement cardinality");
  if (accesses.empty())
    return refuse(loopReference,
                  StructuredScopRefusalKind::AccessRelationProofNotEstablished);
  const std::uint64_t elementWidth = accesses.front().view.elementBytes;
  if (llvm::any_of(accesses, [&](const AccessProof &access) {
        return access.view.elementBytes != elementWidth;
      }))
    return refuse(loopReference,
                  StructuredScopRefusalKind::HeterogeneousElementWidth);

  mlir::AliasAnalysis aliases(providerModule);
  for (std::size_t lhs = 0; lhs != accesses.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs != accesses.size(); ++rhs) {
      if (!accesses[lhs].writes && !accesses[rhs].writes)
        continue;
      if (accesses[lhs].memref != accesses[rhs].memref &&
          !aliases.alias(accesses[lhs].memref, accesses[rhs].memref).isNo())
        return refuse(loopReference,
                      StructuredScopRefusalKind::AliasProofNotEstablished);
    }
  }

  const unsigned dependenceDepth =
      mlir::affine::getNestingDepth(loop.getOperation()) + 1;
  for (const AccessProof &source : accesses) {
    for (const AccessProof &destination : accesses) {
      if (!source.writes && !destination.writes)
        continue;
      const mlir::affine::DependenceResult dependence =
          mlir::affine::checkMemrefAccessDependence(
              source.access, destination.access, dependenceDepth);
      if (dependence.value == mlir::affine::DependenceResult::Failure)
        return refuse(loopReference,
                      StructuredScopRefusalKind::DependenceProofNotEstablished);
      if (dependence.value == mlir::affine::DependenceResult::HasDependence)
        return refuse(loopReference,
                      StructuredScopRefusalKind::LoopCarriedMemoryDependence);
    }
  }

  llvm::SmallVector<mlir::affine::LoopReduction> reductions;
  if (!mlir::affine::isLoopParallel(loop, &reductions))
    return refuse(loopReference,
                  StructuredScopRefusalKind::UnsupportedReduction);
  auto reductionSchedule = analyzeReductions(loop, reductions);
  if (!reductionSchedule)
    return reductionSchedule.takeError();
  if (auto *refusal =
          std::get_if<StructuredScopRefusalKind>(&*reductionSchedule))
    return refuse(loopReference, *refusal);
  if (!reductions.empty()) {
    if (accesses.size() != 1 || accesses.front().writes ||
        computes.size() != 1 ||
        !llvm::isa_and_nonnull<mlir::affine::AffineLoadOp>(
            reductions.front().value.getDefiningOp()))
      return refuse(loopReference,
                    StructuredScopRefusalKind::UnsupportedReduction);
    const StructuredScopComputeView &combiner = computes.front();
    const std::uint64_t loadStatement = accesses.front().view.statementOrdinal;
    if (combiner.operandStatements.size() != 2 ||
        llvm::count_if(combiner.operandStatements,
                       [](const std::optional<std::uint64_t> &operand) {
                         return !operand;
                       }) != 1 ||
        llvm::count(combiner.operandStatements,
                    std::optional<std::uint64_t>(loadStatement)) != 1)
      return refuse(loopReference,
                    StructuredScopRefusalKind::UnsupportedReduction);
    if (sourceLoop->getNumResults() != 1 ||
        !sourceLoop->getResult(0).hasOneUse())
      return refuse(loopReference,
                    StructuredScopRefusalKind::UnsupportedReduction);
    auto returned = llvm::dyn_cast<mlir::func::ReturnOp>(
        *sourceLoop->getResult(0).user_begin());
    if (!returned || returned.getNumOperands() != 1 ||
        returned.getOperand(0) != sourceLoop->getResult(0))
      return refuse(loopReference,
                    StructuredScopRefusalKind::UnsupportedReduction);
  }

  ExactStructuredScopView result(loopReference);
  mlir::Operation *symbolOwner = sourceLoop->getParentOp();
  while (symbolOwner && !symbolOwner->getAttrOfType<mlir::StringAttr>(
                            mlir::SymbolTable::getSymbolAttrName()))
    symbolOwner = symbolOwner->getParentOp();
  if (!symbolOwner)
    return invalid("exact SCoP has no symbol owner");
  result.ownerSymbol = symbolOwner
                           ->getAttrOfType<mlir::StringAttr>(
                               mlir::SymbolTable::getSymbolAttrName())
                           .getValue()
                           .str();
  bool foundSourceLoop = false;
  symbolOwner->walk([&](mlir::Operation *candidate) {
    if (foundSourceLoop ||
        !llvm::isa<mlir::scf::ForOp, mlir::affine::AffineForOp>(candidate))
      return mlir::WalkResult::advance();
    if (candidate == sourceLoop) {
      foundSourceLoop = true;
      return mlir::WalkResult::interrupt();
    }
    ++result.loopOrdinalInOwner;
    return mlir::WalkResult::advance();
  });
  if (!foundSourceLoop)
    return invalid("exact SCoP loop is outside its symbol owner");
  result.statementCount = sourceStatementOps.size();
  result.parameterCount = domain.getNumSymbolVars();
  result.domainConstraintCount = domain.getNumConstraints();
  result.reductionSchedule =
      std::get<StructuredReductionSchedule>(*reductionSchedule);
  result.reductionCount = reductions.size();
  if (!reductions.empty())
    result.reductionKind = reductions.front().kind;
  result.minimumAlignmentBytes = std::numeric_limits<std::uint64_t>::max();
  result.maximumElementBytes = elementWidth;
  for (AccessProof &access : accesses) {
    result.minimumAlignmentBytes =
        std::min(result.minimumAlignmentBytes, access.view.alignmentBytes);
    result.accesses.push_back(std::move(access.view));
  }
  result.computes = std::move(computes);
  if (std::optional<llvm::APInt> tripCount = loop.getStaticTripCount()) {
    if (tripCount->getActiveBits() <= 64)
      result.constantTripCount = tripCount->getZExtValue();
  }
  return StructuredScopAnalysisOutcome(std::move(result));
}

} // namespace

llvm::Expected<mlir::affine::AffineForOp>
projectExactStructuredScopToAffine(mlir::Operation *operation) {
  if (auto affine =
          llvm::dyn_cast_or_null<mlir::affine::AffineForOp>(operation))
    return affine;
  auto loop = llvm::dyn_cast_or_null<mlir::scf::ForOp>(operation);
  if (!loop)
    return invalid("SCoP root is not a supported structured loop");

  mlir::Block *block = loop->getBlock();
  mlir::Operation *predecessor = loop->getPrevNode();
  mlir::Operation *successor = loop->getNextNode();
  mlir::RewritePatternSet patterns(loop.getContext());
  mlir::populateSCFToAffineConversionPatterns(patterns);
  mlir::GreedyRewriteConfig config;
  config.setScope(loop->getParentRegion())
      .setStrictness(mlir::GreedyRewriteStrictness::ExistingAndNewOps)
      .setRegionSimplificationLevel(mlir::GreedySimplifyRegionLevel::Disabled);
  mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  if (mlir::failed(mlir::applyOpPatternsGreedily(
          llvm::ArrayRef<mlir::Operation *>{loop.getOperation()},
          frozenPatterns, config)))
    return invalid("SCF provider rejected the selected loop");
  auto projected = findProjectedLoop(block, predecessor, successor);
  if (!projected)
    return projected.takeError();

  llvm::SmallVector<mlir::Operation *> accesses;
  for (mlir::Operation &candidate :
       projected->getBody()->without_terminator()) {
    llvm::SmallVector<mlir::Value> indices;
    if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(candidate))
      indices.append(load.getIndices().begin(), load.getIndices().end());
    else if (auto store = llvm::dyn_cast<mlir::memref::StoreOp>(candidate))
      indices.append(store.getIndices().begin(), store.getIndices().end());
    else
      continue;
    if (indices.size() == 1 && indices.front() == projected->getInductionVar())
      accesses.push_back(&candidate);
  }
  for (mlir::Operation *candidate : accesses) {
    mlir::OpBuilder builder(candidate);
    if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(candidate)) {
      auto raised = mlir::affine::AffineLoadOp::create(
          builder, load.getLoc(), load.getMemref(), load.getIndices());
      load.getResult().replaceAllUsesWith(raised.getResult());
      load.erase();
      continue;
    }
    auto store = llvm::cast<mlir::memref::StoreOp>(candidate);
    mlir::affine::AffineStoreOp::create(builder, store.getLoc(),
                                        store.getValue(), store.getMemref(),
                                        store.getIndices());
    store.erase();
  }
  if (mlir::failed(mlir::verify(*projected)))
    return invalid("SCF provider projection does not verify as Affine");
  return *projected;
}

llvm::Expected<StructuredScopAnalysisOutcome>
analyzeExactStructuredScop(const StructuredProgramCandidate &parent,
                           const StructuredEntityRef &loopReference) {
  if (loopReference.kind != StructuredEntityKind::Operation)
    return invalid("SCoP root does not reference an operation");
  auto candidateView = parent.view();
  if (!candidateView)
    return candidateView.takeError();
  auto entity = candidateView->resolve(loopReference);
  if (!entity)
    return entity.takeError();
  mlir::Operation *sourceLoop = entity->operation;
  if (!llvm::isa_and_nonnull<mlir::scf::ForOp, mlir::affine::AffineForOp>(
          sourceLoop))
    return refuse(loopReference, StructuredScopRefusalKind::NotAffineLoop);
  mlir::Region &sourceRegion = sourceLoop->getRegion(0);
  for (mlir::Operation &operation : sourceRegion.front().without_terminator()) {
    mlir::MemRefType type;
    if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(operation))
      type = load.getMemRefType();
    else if (auto store = llvm::dyn_cast<mlir::memref::StoreOp>(operation))
      type = store.getMemRefType();
    else if (auto load = llvm::dyn_cast<mlir::affine::AffineLoadOp>(operation))
      type = load.getMemRefType();
    else if (auto store =
                 llvm::dyn_cast<mlir::affine::AffineStoreOp>(operation))
      type = store.getMemRefType();
    if (type)
      if (auto refusal = physicalLayoutRefusal(type))
        return refuse(loopReference, *refusal);
  }

  mlir::IRMapping mapping;
  auto privateClone =
      cloneStructuredProgramWithSourceLocations(parent, {}, mapping);
  if (!privateClone)
    return privateClone.takeError();
  mlir::Operation *clonedLoop = mapping.lookupOrNull(sourceLoop);
  if (!clonedLoop)
    return invalid("selected loop was not mapped into the provider clone");
  auto projected = projectExactStructuredScopToAffine(clonedLoop);
  if (!projected) {
    llvm::consumeError(projected.takeError());
    return refuse(loopReference,
                  StructuredScopRefusalKind::ProviderMaterializationRejected);
  }
  return analyzeProjectedScop(loopReference, sourceLoop, *projected,
                              privateClone->get());
}

} // namespace loom::frontend
