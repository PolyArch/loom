#include "Frontend/Compilation/StructuredScop.h"

#include "StructuredPolyhedralProvider.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/OperationSchema.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
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

constexpr std::uint64_t maximumDependenceQueries = 65'536;

StructuredScopAnalysisOutcome refuse(const StructuredEntityRef &loop,
                                     StructuredScopRefusalKind kind,
                                     std::uint64_t dependenceQueryCount = 0) {
  return StructuredScopRefusal{loop, kind, dependenceQueryCount};
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
    if (!dataflow::isCanonicalDataflowActor(
            sourceOperation, dataflow::CanonicalDataflowActorKind::Compute) ||
        !dataflow::isCanonicalDataflowActor(
            &operation, dataflow::CanonicalDataflowActorKind::Compute))
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
  if (sourceStatementOps.size() > detail::maximumPinnedIslStatementCount)
    return refuse(loopReference,
                  StructuredScopRefusalKind::ProviderDomainNotAdmitted);
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
  struct MemoryDependence final {
    std::uint64_t sourceStatementOrdinal = 0;
    std::uint64_t destinationStatementOrdinal = 0;
    mlir::affine::FlatAffineValueConstraints relation;
  };
  std::vector<MemoryDependence> memoryDependences;
  std::uint64_t dependenceQueryCount = 0;
  const auto refuseAfterQueries = [&](StructuredScopRefusalKind kind) {
    return refuse(loopReference, kind, dependenceQueryCount);
  };
  const auto consumeDependenceQuery = [&]() -> bool {
    if (dependenceQueryCount == maximumDependenceQueries)
      return false;
    ++dependenceQueryCount;
    return true;
  };
  for (const AccessProof &source : accesses) {
    for (const AccessProof &destination : accesses) {
      if (!source.writes && !destination.writes)
        continue;
      if (!consumeDependenceQuery())
        return refuseAfterQueries(
            StructuredScopRefusalKind::ProviderScheduleBudgetExhausted);
      const mlir::affine::DependenceResult dependence =
          mlir::affine::checkMemrefAccessDependence(
              source.access, destination.access, dependenceDepth);
      if (dependence.value == mlir::affine::DependenceResult::Failure)
        return refuseAfterQueries(
            StructuredScopRefusalKind::DependenceProofNotEstablished);
      if (dependence.value == mlir::affine::DependenceResult::HasDependence)
        return refuseAfterQueries(
            StructuredScopRefusalKind::LoopCarriedMemoryDependence);

      mlir::affine::FlatAffineValueConstraints statementRelation;
      if (!consumeDependenceQuery())
        return refuseAfterQueries(
            StructuredScopRefusalKind::ProviderScheduleBudgetExhausted);
      const mlir::affine::DependenceResult statementDependence =
          mlir::affine::checkMemrefAccessDependence(
              source.access, destination.access, dependenceDepth + 1,
              &statementRelation);
      if (statementDependence.value == mlir::affine::DependenceResult::Failure)
        return refuseAfterQueries(
            StructuredScopRefusalKind::DependenceProofNotEstablished);
      if (statementDependence.value ==
          mlir::affine::DependenceResult::HasDependence)
        memoryDependences.push_back({source.view.statementOrdinal,
                                     destination.view.statementOrdinal,
                                     std::move(statementRelation)});
    }
  }

  llvm::SmallVector<mlir::affine::LoopReduction> reductions;
  if (!mlir::affine::isLoopParallel(loop, &reductions))
    return refuseAfterQueries(StructuredScopRefusalKind::UnsupportedReduction);
  auto reductionSchedule = analyzeReductions(loop, reductions);
  if (!reductionSchedule)
    return reductionSchedule.takeError();
  if (auto *refusal =
          std::get_if<StructuredScopRefusalKind>(&*reductionSchedule))
    return refuseAfterQueries(*refusal);
  if (!reductions.empty()) {
    if (accesses.size() != 1 || accesses.front().writes ||
        computes.size() != 1 ||
        !llvm::isa_and_nonnull<mlir::affine::AffineLoadOp>(
            reductions.front().value.getDefiningOp()))
      return refuseAfterQueries(
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
      return refuseAfterQueries(
          StructuredScopRefusalKind::UnsupportedReduction);
    if (sourceLoop->getNumResults() != 1 ||
        !sourceLoop->getResult(0).hasOneUse())
      return refuseAfterQueries(
          StructuredScopRefusalKind::UnsupportedReduction);
    auto returned = llvm::dyn_cast<mlir::func::ReturnOp>(
        *sourceLoop->getResult(0).user_begin());
    if (!returned || returned.getNumOperands() != 1 ||
        returned.getOperand(0) != sourceLoop->getResult(0))
      return refuseAfterQueries(
          StructuredScopRefusalKind::UnsupportedReduction);
  }

  std::vector<mlir::affine::FlatAffineValueConstraints> providerDomains(
      sourceStatementOps.size(), domain);
  std::vector<detail::PolyhedralStatementDomain> providerStatements;
  providerStatements.reserve(providerDomains.size());
  for (auto [ordinal, statementDomain] : llvm::enumerate(providerDomains))
    providerStatements.push_back({ordinal, &statementDomain});
  std::vector<detail::PolyhedralDependenceRelation> providerDependences;
  providerDependences.reserve(memoryDependences.size());
  for (const MemoryDependence &dependence : memoryDependences)
    providerDependences.push_back(
        {dependence.sourceStatementOrdinal,
         dependence.destinationStatementOrdinal, domain.getNumDimVars(),
         domain.getNumDimVars(), &dependence.relation});
  const auto appendPrecedence = [&](std::uint64_t source,
                                    std::uint64_t destination) {
    if (source == destination)
      return;
    const bool duplicate =
        llvm::any_of(providerDependences, [&](const auto &dependence) {
          return dependence.sourceStatementOrdinal == source &&
                 dependence.destinationStatementOrdinal == destination &&
                 dependence.relation == nullptr;
        });
    if (!duplicate)
      providerDependences.push_back({source, destination,
                                     domain.getNumDimVars(),
                                     domain.getNumDimVars(), nullptr});
  };
  for (const StructuredScopComputeView &compute : computes)
    for (const std::optional<std::uint64_t> &operand :
         compute.operandStatements)
      if (operand)
        appendPrecedence(*operand, compute.statementOrdinal);
  for (const AccessProof &access : accesses)
    if (access.view.storedStatementOrdinal)
      appendPrecedence(*access.view.storedStatementOrdinal,
                       access.view.statementOrdinal);
  auto providerOutcome =
      detail::computePinnedIslSchedule(providerStatements, providerDependences);
  if (!providerOutcome)
    return providerOutcome.takeError();
  if (auto *providerRefusal =
          std::get_if<detail::PolyhedralScheduleProviderRefusalKind>(
              &*providerOutcome)) {
    StructuredScopRefusalKind refusal =
        StructuredScopRefusalKind::ProviderScheduleNotEstablished;
    switch (*providerRefusal) {
    case detail::PolyhedralScheduleProviderRefusalKind::DomainNotAdmitted:
      refusal = StructuredScopRefusalKind::ProviderDomainNotAdmitted;
      break;
    case detail::PolyhedralScheduleProviderRefusalKind::ScheduleNotEstablished:
      break;
    case detail::PolyhedralScheduleProviderRefusalKind::
        OperationBudgetExhausted:
      refusal = StructuredScopRefusalKind::ProviderScheduleBudgetExhausted;
      break;
    }
    return refuseAfterQueries(refusal);
  }
  detail::PolyhedralScheduleProviderView &providerSchedule =
      std::get<detail::PolyhedralScheduleProviderView>(*providerOutcome);
  if (providerSchedule.statementSchedules.size() != sourceStatementOps.size())
    return invalid("Polly/ISL schedule changed statement cardinality");

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
  result.dependenceQueryCount = dependenceQueryCount;
  result.reductionSchedule =
      std::get<StructuredReductionSchedule>(*reductionSchedule);
  result.reductionCount = reductions.size();
  if (!reductions.empty())
    result.reductionKind = reductions.front().kind;
  result.polyhedralSchedule = {StructuredPolyhedralProviderKind::PinnedPollyIsl,
                               providerSchedule.form,
                               providerSchedule.parameterCount,
                               providerDependences.size(),
                               providerSchedule.scheduleBandCount,
                               providerSchedule.scheduleDimensionCount,
                               providerSchedule.coincidentDimensionCount,
                               std::move(providerSchedule.statementSchedules)};
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
  mlir::affine::AffineForOp projected =
      llvm::dyn_cast_or_null<mlir::affine::AffineForOp>(operation);
  auto loop = llvm::dyn_cast_or_null<mlir::scf::ForOp>(operation);
  if (!projected && !loop)
    return invalid("SCoP root is not a supported structured loop");

  if (loop) {
    mlir::Block *block = loop->getBlock();
    mlir::Operation *predecessor = loop->getPrevNode();
    mlir::Operation *successor = loop->getNextNode();
    mlir::RewritePatternSet patterns(loop.getContext());
    mlir::populateSCFToAffineConversionPatterns(patterns);
    mlir::GreedyRewriteConfig config;
    config.setScope(loop->getParentRegion())
        .setStrictness(mlir::GreedyRewriteStrictness::ExistingAndNewOps)
        .setRegionSimplificationLevel(
            mlir::GreedySimplifyRegionLevel::Disabled);
    mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (mlir::failed(mlir::applyOpPatternsGreedily(
            llvm::ArrayRef<mlir::Operation *>{loop.getOperation()},
            frozenPatterns, config)))
      return invalid("SCF provider rejected the selected loop");
    auto converted = findProjectedLoop(block, predecessor, successor);
    if (!converted)
      return converted.takeError();
    projected = *converted;
  }

  llvm::SmallVector<mlir::Operation *> accesses;
  projected->walk([&](mlir::Operation *candidate) {
    mlir::ValueRange indices;
    if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(candidate))
      indices = load.getIndices();
    else if (auto store = llvm::dyn_cast<mlir::memref::StoreOp>(candidate))
      indices = store.getIndices();
    else
      return;
    if (llvm::all_of(indices, [](mlir::Value index) {
          return mlir::affine::isValidDim(index);
        }))
      accesses.push_back(candidate);
  });
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
  if (mlir::failed(mlir::verify(projected)))
    return invalid("SCF provider projection does not verify as Affine");
  return projected;
}

namespace {

enum class PolyhedralStatementClass : std::uint32_t {
  Access = 0,
  Actor = 1,
  AffineSupport = 2,
};

struct PolyhedralLoopRecord final {
  mlir::Operation *operation = nullptr;
  mlir::Value inductionVariable;
  std::optional<std::size_t> parent;
};

struct PolyhedralStatementRecord final {
  mlir::Operation *operation = nullptr;
  PolyhedralStatementClass statementClass = PolyhedralStatementClass::Actor;
};

struct PolyhedralStructure final {
  std::vector<PolyhedralLoopRecord> loops;
  std::vector<PolyhedralStatementRecord> statements;
  std::uint64_t maximumLoopDepth = 0;
  bool imperfectNest = false;
};

bool isAccessOperation(mlir::Operation *operation) {
  return llvm::isa<mlir::memref::LoadOp, mlir::memref::StoreOp,
                   mlir::affine::AffineLoadOp, mlir::affine::AffineStoreOp>(
      operation);
}

bool isAffineSupportOperation(mlir::Operation *operation) {
  return llvm::isa<mlir::affine::AffineApplyOp, mlir::affine::AffineMinOp,
                   mlir::affine::AffineMaxOp>(operation);
}

mlir::Value loopInductionVariable(mlir::Operation *operation) {
  if (auto loop = llvm::dyn_cast<mlir::scf::ForOp>(operation))
    return loop.getInductionVar();
  return llvm::cast<mlir::affine::AffineForOp>(operation).getInductionVar();
}

mlir::Block *loopBody(mlir::Operation *operation) {
  if (auto loop = llvm::dyn_cast<mlir::scf::ForOp>(operation))
    return loop.getBody();
  return llvm::cast<mlir::affine::AffineForOp>(operation).getBody();
}

std::optional<StructuredScopRefusalKind>
collectPolyhedralLoop(mlir::Operation *loop, std::optional<std::size_t> parent,
                      PolyhedralStructure &structure) {
  if (auto scf = llvm::dyn_cast<mlir::scf::ForOp>(loop)) {
    if (!scf.getInitArgs().empty())
      return StructuredScopRefusalKind::UnsupportedReduction;
    const std::optional<std::int64_t> step =
        mlir::getConstantIntValue(scf.getStep());
    if (!step || *step != 1)
      return StructuredScopRefusalKind::ProviderDomainNotAdmitted;
  } else if (auto affine = llvm::dyn_cast<mlir::affine::AffineForOp>(loop)) {
    if (!affine.getInits().empty())
      return StructuredScopRefusalKind::UnsupportedReduction;
    if (affine.getStepAsInt() != 1)
      return StructuredScopRefusalKind::ProviderDomainNotAdmitted;
  } else {
    return StructuredScopRefusalKind::NotAffineLoop;
  }

  const std::size_t ordinal = structure.loops.size();
  structure.loops.push_back({loop, loopInductionVariable(loop), parent});
  std::uint64_t depth = 1;
  for (std::optional<std::size_t> cursor = parent; cursor;
       cursor = structure.loops[*cursor].parent)
    ++depth;
  structure.maximumLoopDepth = std::max(structure.maximumLoopDepth, depth);

  std::size_t nestedLoops = 0;
  std::size_t directStatements = 0;
  for (mlir::Operation &operation : loopBody(loop)->without_terminator()) {
    if (llvm::isa<mlir::scf::ForOp, mlir::affine::AffineForOp>(operation)) {
      ++nestedLoops;
      if (auto refusal = collectPolyhedralLoop(&operation, ordinal, structure))
        return refusal;
      continue;
    }
    if (operation.getNumRegions() != 0 ||
        llvm::isa<mlir::scf::IfOp, mlir::scf::WhileOp, mlir::affine::AffineIfOp,
                  mlir::affine::AffineParallelOp>(operation))
      return StructuredScopRefusalKind::NestedControl;
    if (isAccessOperation(&operation)) {
      ++directStatements;
      structure.statements.push_back(
          {&operation, PolyhedralStatementClass::Access});
      continue;
    }
    if (isAffineSupportOperation(&operation)) {
      ++directStatements;
      structure.statements.push_back(
          {&operation, PolyhedralStatementClass::AffineSupport});
      continue;
    }
    const std::optional<dataflow::CanonicalDataflowActorKind> actorKind =
        dataflow::classifyCanonicalDataflowActor(&operation);
    if (actorKind == dataflow::CanonicalDataflowActorKind::Compute) {
      ++directStatements;
      structure.statements.push_back(
          {&operation, PolyhedralStatementClass::Actor});
      continue;
    }
    if (actorKind)
      return *actorKind == dataflow::CanonicalDataflowActorKind::Memory
                 ? StructuredScopRefusalKind::UnsupportedEffect
                 : StructuredScopRefusalKind::UnsupportedOperation;
    return mlir::isMemoryEffectFree(&operation)
               ? StructuredScopRefusalKind::UnsupportedOperation
               : StructuredScopRefusalKind::UnsupportedEffect;
  }
  if (nestedLoops != 0 && (nestedLoops != 1 || directStatements != 0))
    structure.imperfectNest = true;
  return std::nullopt;
}

llvm::Expected<std::optional<StructuredScopRefusalKind>>
comparePolyhedralStructures(const PolyhedralStructure &source,
                            const PolyhedralStructure &projected) {
  if (source.loops.size() != projected.loops.size() ||
      source.statements.size() != projected.statements.size())
    return StructuredScopRefusalKind::ProviderMaterializationRejected;
  for (auto [sourceLoop, projectedLoop] :
       llvm::zip(source.loops, projected.loops))
    if (sourceLoop.parent != projectedLoop.parent)
      return StructuredScopRefusalKind::ProviderMaterializationRejected;
  for (auto [sourceStatement, projectedStatement] :
       llvm::zip(source.statements, projected.statements)) {
    if (sourceStatement.statementClass != projectedStatement.statementClass ||
        sourceStatement.operation->getNumResults() !=
            projectedStatement.operation->getNumResults())
      return StructuredScopRefusalKind::ProviderMaterializationRejected;
    if (sourceStatement.statementClass != PolyhedralStatementClass::Actor)
      continue;
    auto sourceProjection = dataflow::projectRegisteredActorSchemaProjection(
        sourceStatement.operation);
    if (!sourceProjection)
      return sourceProjection.takeError();
    auto projectedProjection = dataflow::projectRegisteredActorSchemaProjection(
        projectedStatement.operation);
    if (!projectedProjection)
      return projectedProjection.takeError();
    if (sourceProjection->schema != projectedProjection->schema ||
        !(sourceProjection->payload == projectedProjection->payload))
      return StructuredScopRefusalKind::ProviderMaterializationRejected;
  }
  return std::nullopt;
}

template <typename Relation>
void freezeConstraintRows(
    const Relation &relation,
    std::vector<StructuredPolyhedralConstraintView> &out) {
  out.reserve(relation.getNumConstraints());
  for (unsigned index = 0; index != relation.getNumInequalities(); ++index) {
    auto row = relation.getInequality64(index);
    out.push_back({StructuredPolyhedralConstraintKind::Inequality,
                   std::vector<std::int64_t>(row.begin(), row.end())});
  }
  for (unsigned index = 0; index != relation.getNumEqualities(); ++index) {
    auto row = relation.getEquality64(index);
    out.push_back({StructuredPolyhedralConstraintKind::Equality,
                   std::vector<std::int64_t>(row.begin(), row.end())});
  }
}

llvm::Expected<StructuredEntityRef> valueReference(
    mlir::Value value,
    const llvm::DenseMap<mlir::Value, StructuredEntityRef> &references) {
  auto found = references.find(value);
  if (found == references.end())
    return invalid("polyhedral value has no exact source entity");
  return found->second;
}

llvm::Expected<StructuredPolyhedralSetView>
freezeSet(const mlir::affine::FlatAffineValueConstraints &domain,
          const llvm::DenseMap<mlir::Value, StructuredEntityRef> &references) {
  if (domain.getNumLocalVars() != 0)
    return invalid("polyhedral set contains an unadmitted local variable");
  StructuredPolyhedralSetView view;
  view.dimensions.reserve(domain.getNumDimVars());
  for (unsigned index = 0; index != domain.getNumDimVars(); ++index) {
    if (!domain.hasValue(index))
      return invalid("polyhedral dimension lost its source identity");
    auto reference = valueReference(domain.getValue(index), references);
    if (!reference)
      return reference.takeError();
    view.dimensions.push_back(*reference);
  }
  view.parameters.reserve(domain.getNumSymbolVars());
  for (unsigned index = 0; index != domain.getNumSymbolVars(); ++index) {
    const unsigned position = domain.getNumDimVars() + index;
    if (!domain.hasValue(position))
      return invalid("polyhedral parameter lost its source identity");
    auto reference = valueReference(domain.getValue(position), references);
    if (!reference)
      return reference.takeError();
    view.parameters.push_back(*reference);
  }
  freezeConstraintRows(domain, view.constraints);
  return view;
}

llvm::Expected<StructuredPolyhedralRelationView> freezeRelation(
    const mlir::affine::FlatAffineValueConstraints &relation,
    std::uint64_t sourceDimensions, std::uint64_t destinationDimensions,
    const llvm::DenseMap<mlir::Value, StructuredEntityRef> &references) {
  if (relation.getNumLocalVars() != 0 ||
      relation.getNumDimVars() != sourceDimensions + destinationDimensions)
    return invalid("polyhedral relation has an unadmitted variable space");
  StructuredPolyhedralRelationView view;
  view.sourceDimensionCount = sourceDimensions;
  view.destinationDimensionCount = destinationDimensions;
  for (unsigned index = 0; index != relation.getNumSymbolVars(); ++index) {
    const unsigned position = relation.getNumDimVars() + index;
    if (!relation.hasValue(position))
      return invalid("polyhedral relation parameter lost its source identity");
    auto reference = valueReference(relation.getValue(position), references);
    if (!reference)
      return reference.takeError();
    view.parameters.push_back(*reference);
  }
  freezeConstraintRows(relation, view.constraints);
  return view;
}

llvm::Expected<StructuredPolyhedralRelationView> freezeAccessRelation(
    const mlir::presburger::IntegerRelation &relation,
    const llvm::DenseMap<mlir::Value, StructuredEntityRef> &references) {
  if (relation.getNumLocalVars() != 0)
    return invalid("polyhedral access contains an unadmitted local variable");
  StructuredPolyhedralRelationView view;
  view.sourceDimensionCount = relation.getNumDomainVars();
  view.destinationDimensionCount = relation.getNumRangeVars();
  for (unsigned index = 0; index != relation.getNumSymbolVars(); ++index) {
    const mlir::presburger::Identifier id =
        relation.getSpace().getId(mlir::presburger::VarKind::Symbol, index);
    if (!id.hasValue())
      return invalid("polyhedral access parameter lost its source identity");
    auto reference = valueReference(id.getValue<mlir::Value>(), references);
    if (!reference)
      return reference.takeError();
    view.parameters.push_back(*reference);
  }
  freezeConstraintRows(relation, view.constraints);
  return view;
}

mlir::Value accessMemory(mlir::Operation *operation) {
  if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(operation))
    return load.getMemref();
  if (auto store = llvm::dyn_cast<mlir::memref::StoreOp>(operation))
    return store.getMemref();
  if (auto load = llvm::dyn_cast<mlir::affine::AffineLoadOp>(operation))
    return load.getMemRef();
  return llvm::cast<mlir::affine::AffineStoreOp>(operation).getMemRef();
}

bool accessWrites(mlir::Operation *operation) {
  return llvm::isa<mlir::memref::StoreOp, mlir::affine::AffineStoreOp>(
      operation);
}

StructuredPolyhedralDependenceKind dependenceKind(bool sourceWrites,
                                                  bool destinationWrites) {
  if (sourceWrites && destinationWrites)
    return StructuredPolyhedralDependenceKind::WriteAfterWrite;
  if (sourceWrites)
    return StructuredPolyhedralDependenceKind::ReadAfterWrite;
  return StructuredPolyhedralDependenceKind::WriteAfterRead;
}

StructuredPolyhedralScopAnalysisOutcome
refusePolyhedral(const StructuredEntityRef &loop,
                 StructuredScopRefusalKind kind,
                 std::uint64_t dependenceQueryCount = 0) {
  return StructuredScopRefusal{loop, kind, dependenceQueryCount};
}

bool valueDefinedInside(mlir::Value value, mlir::Operation *root) {
  mlir::Region *region = value.getParentRegion();
  if (!region || root->getNumRegions() == 0)
    return false;
  mlir::Region &rootRegion = root->getRegion(0);
  return region == &rootRegion || rootRegion.isAncestor(region);
}

} // namespace

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

llvm::Expected<StructuredPolyhedralScopAnalysisOutcome>
analyzeStructuredPolyhedralScop(const StructuredProgramCandidate &parent,
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
    return refusePolyhedral(loopReference,
                            StructuredScopRefusalKind::NotAffineLoop);
  if (sourceLoop->getParentOfType<mlir::scf::ForOp>() ||
      sourceLoop->getParentOfType<mlir::affine::AffineForOp>() ||
      sourceLoop->getParentOfType<mlir::scf::IfOp>() ||
      sourceLoop->getParentOfType<mlir::affine::AffineIfOp>())
    return refusePolyhedral(loopReference,
                            StructuredScopRefusalKind::NestedAffineRoot);

  PolyhedralStructure sourceStructure;
  if (auto refusal =
          collectPolyhedralLoop(sourceLoop, std::nullopt, sourceStructure))
    return refusePolyhedral(loopReference, *refusal);
  if (sourceStructure.statements.empty() ||
      sourceStructure.statements.size() >
          detail::maximumPinnedIslStatementCount)
    return refusePolyhedral(
        loopReference, StructuredScopRefusalKind::ProviderDomainNotAdmitted);

  llvm::DenseMap<mlir::Operation *, StructuredEntityRef> operationReferences;
  llvm::DenseMap<mlir::Value, StructuredEntityRef> sourceValueReferences;
  for (const StructuredEntity &candidate :
       candidateView->entities(StructuredEntityKind::Operation))
    operationReferences.try_emplace(candidate.operation, candidate.reference);
  for (const StructuredEntity &candidate :
       candidateView->entities(StructuredEntityKind::Value))
    sourceValueReferences.try_emplace(candidate.value, candidate.reference);

  mlir::IRMapping mapping;
  auto privateClone =
      cloneStructuredProgramWithSourceLocations(parent, {}, mapping);
  if (!privateClone)
    return privateClone.takeError();
  llvm::DenseMap<mlir::Value, StructuredEntityRef> projectedValueReferences;
  for (const auto &[source, reference] : sourceValueReferences)
    if (mlir::Value projected = mapping.lookupOrNull(source))
      projectedValueReferences.try_emplace(projected, reference);
  mlir::Operation *clonedLoop = mapping.lookupOrNull(sourceLoop);
  if (!clonedLoop)
    return invalid("selected loop was not mapped into the provider clone");
  auto projectedLoop = projectExactStructuredScopToAffine(clonedLoop);
  if (!projectedLoop) {
    llvm::consumeError(projectedLoop.takeError());
    return refusePolyhedral(
        loopReference,
        StructuredScopRefusalKind::ProviderMaterializationRejected);
  }

  PolyhedralStructure projectedStructure;
  if (auto refusal = collectPolyhedralLoop(projectedLoop->getOperation(),
                                           std::nullopt, projectedStructure))
    return refusePolyhedral(loopReference, *refusal);
  auto structureComparison =
      comparePolyhedralStructures(sourceStructure, projectedStructure);
  if (!structureComparison)
    return structureComparison.takeError();
  if (*structureComparison)
    return refusePolyhedral(loopReference, **structureComparison);

  for (auto [source, projected] :
       llvm::zip(sourceStructure.loops, projectedStructure.loops)) {
    auto reference = sourceValueReferences.find(source.inductionVariable);
    if (reference == sourceValueReferences.end())
      return invalid("source loop induction variable has no entity");
    projectedValueReferences.insert_or_assign(projected.inductionVariable,
                                              reference->second);
  }
  for (auto [source, projected] :
       llvm::zip(sourceStructure.statements, projectedStructure.statements)) {
    for (auto [sourceResult, projectedResult] :
         llvm::zip(source.operation->getResults(),
                   projected.operation->getResults())) {
      auto reference = sourceValueReferences.find(sourceResult);
      if (reference == sourceValueReferences.end())
        return invalid("source statement result has no entity");
      projectedValueReferences.insert_or_assign(projectedResult,
                                                reference->second);
    }
  }

  struct StatementProof final {
    mlir::Operation *source = nullptr;
    mlir::Operation *projected = nullptr;
    mlir::affine::FlatAffineValueConstraints domain;
  };
  std::vector<StatementProof> statements;
  statements.reserve(sourceStructure.statements.size());
  StructuredPolyhedralScopView result(loopReference);
  result.loopCount = sourceStructure.loops.size();
  result.maximumLoopDepth = sourceStructure.maximumLoopDepth;
  result.imperfectNest = sourceStructure.imperfectNest;
  result.statements.reserve(sourceStructure.statements.size());
  for (auto [source, projected] :
       llvm::zip(sourceStructure.statements, projectedStructure.statements)) {
    llvm::SmallVector<mlir::Operation *> enclosing;
    mlir::affine::getEnclosingAffineOps(*projected.operation, &enclosing);
    mlir::affine::FlatAffineValueConstraints domain;
    if (mlir::failed(mlir::affine::getIndexSet(enclosing, &domain)))
      return refusePolyhedral(
          loopReference, StructuredScopRefusalKind::DomainProofNotEstablished);
    if (domain.getNumLocalVars() != 0)
      return refusePolyhedral(
          loopReference, StructuredScopRefusalKind::ProviderDomainNotAdmitted);
    auto sourceReference = operationReferences.find(source.operation);
    if (sourceReference == operationReferences.end())
      return invalid("source statement has no operation entity");
    auto frozen = freezeSet(domain, projectedValueReferences);
    if (!frozen)
      return frozen.takeError();
    result.statements.push_back({sourceReference->second, std::move(*frozen)});
    statements.push_back(
        {source.operation, projected.operation, std::move(domain)});
  }

  struct GeneralAccessProof final {
    mlir::affine::MemRefAccess access;
    mlir::Value memory;
    bool writes = false;
    std::uint64_t statementOrdinal = 0;
  };
  std::vector<GeneralAccessProof> accesses;
  for (auto [ordinal, statement] : llvm::enumerate(statements)) {
    if (!isAccessOperation(statement.projected))
      continue;
    if (!llvm::isa<mlir::affine::AffineLoadOp, mlir::affine::AffineStoreOp>(
            statement.projected))
      return refusePolyhedral(
          loopReference,
          StructuredScopRefusalKind::AccessRelationProofNotEstablished);
    mlir::affine::MemRefAccess access(statement.projected);
    mlir::presburger::IntegerRelation relation(
        mlir::presburger::PresburgerSpace::getRelationSpace());
    if (mlir::failed(access.getAccessRelation(relation)))
      return refusePolyhedral(
          loopReference,
          StructuredScopRefusalKind::AccessRelationProofNotEstablished);
    if (relation.getNumLocalVars() != 0 ||
        relation.getNumDomainVars() !=
            statements[ordinal].domain.getNumDimVars())
      return refusePolyhedral(
          loopReference, StructuredScopRefusalKind::ProviderDomainNotAdmitted);
    auto frozenRelation =
        freezeAccessRelation(relation, projectedValueReferences);
    if (!frozenRelation)
      return frozenRelation.takeError();
    auto operationReference = operationReferences.find(statement.source);
    if (operationReference == operationReferences.end())
      return invalid("source access has no operation entity");
    auto memoryReference =
        valueReference(accessMemory(statement.source), sourceValueReferences);
    if (!memoryReference)
      return memoryReference.takeError();
    auto bytes = elementBytes(
        llvm::cast<mlir::MemRefType>(access.memref.getType()).getElementType(),
        statement.projected);
    if (!bytes)
      return bytes.takeError();
    mlir::affine::MemRefRegion footprint(statement.projected->getLoc());
    if (mlir::failed(footprint.compute(statement.projected, 0)))
      return refusePolyhedral(
          loopReference,
          StructuredScopRefusalKind::AccessRelationProofNotEstablished);
    std::optional<std::uint64_t> footprintElements;
    if (std::optional<std::int64_t> upper =
            footprint.getConstantBoundingSizeAndShape()) {
      if (*upper < 0)
        return invalid("polyhedral footprint has a negative upper bound");
      footprintElements = static_cast<std::uint64_t>(*upper);
    }
    const bool writes = accessWrites(statement.projected);
    result.accesses.push_back(
        {operationReference->second, *memoryReference, ordinal,
         writes ? StructuredPolyhedralAccessKind::Write
                : StructuredPolyhedralAccessKind::Read,
         *bytes, std::move(*frozenRelation), footprintElements});
    const mlir::Value projectedMemory = access.memref;
    accesses.push_back({std::move(access), projectedMemory, writes, ordinal});
  }
  if (accesses.empty())
    return refusePolyhedral(
        loopReference,
        StructuredScopRefusalKind::AccessRelationProofNotEstablished);

  mlir::AliasAnalysis aliases(privateClone->get());
  for (std::size_t lhs = 0; lhs != accesses.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs != accesses.size(); ++rhs) {
      if (!accesses[lhs].writes && !accesses[rhs].writes)
        continue;
      if (accesses[lhs].memory != accesses[rhs].memory &&
          !aliases.alias(accesses[lhs].memory, accesses[rhs].memory).isNo())
        return refusePolyhedral(
            loopReference, StructuredScopRefusalKind::AliasProofNotEstablished);
    }
  }

  struct MemoryDependence final {
    StructuredPolyhedralDependenceKind kind =
        StructuredPolyhedralDependenceKind::ReadAfterWrite;
    std::uint64_t source = 0;
    std::uint64_t destination = 0;
    mlir::affine::FlatAffineValueConstraints relation;
  };
  std::vector<MemoryDependence> memoryDependences;
  for (const GeneralAccessProof &source : accesses) {
    for (const GeneralAccessProof &destination : accesses) {
      if ((!source.writes && !destination.writes) ||
          source.memory != destination.memory)
        continue;
      const auto &sourceDomain = statements[source.statementOrdinal].domain;
      const auto &destinationDomain =
          statements[destination.statementOrdinal].domain;
      const unsigned commonLoops = mlir::affine::getNumCommonSurroundingLoops(
          *source.access.opInst, *destination.access.opInst);
      for (unsigned depth = 1; depth <= commonLoops + 1; ++depth) {
        if (result.dependenceQueryCount == maximumDependenceQueries)
          return refusePolyhedral(
              loopReference,
              StructuredScopRefusalKind::ProviderScheduleBudgetExhausted,
              result.dependenceQueryCount);
        ++result.dependenceQueryCount;
        mlir::affine::FlatAffineValueConstraints relation;
        const mlir::affine::DependenceResult dependence =
            mlir::affine::checkMemrefAccessDependence(
                source.access, destination.access, depth, &relation);
        if (dependence.value == mlir::affine::DependenceResult::Failure)
          return refusePolyhedral(
              loopReference,
              StructuredScopRefusalKind::DependenceProofNotEstablished,
              result.dependenceQueryCount);
        if (dependence.value != mlir::affine::DependenceResult::HasDependence)
          continue;
        if (relation.getNumLocalVars() != 0 ||
            relation.getNumDimVars() != sourceDomain.getNumDimVars() +
                                            destinationDomain.getNumDimVars())
          return refusePolyhedral(
              loopReference,
              StructuredScopRefusalKind::ProviderDomainNotAdmitted,
              result.dependenceQueryCount);
        const StructuredPolyhedralDependenceKind kind =
            dependenceKind(source.writes, destination.writes);
        auto frozen = freezeRelation(relation, sourceDomain.getNumDimVars(),
                                     destinationDomain.getNumDimVars(),
                                     projectedValueReferences);
        if (!frozen)
          return frozen.takeError();
        result.dependences.push_back({kind, source.statementOrdinal,
                                      destination.statementOrdinal,
                                      std::move(*frozen)});
        memoryDependences.push_back({kind, source.statementOrdinal,
                                     destination.statementOrdinal,
                                     std::move(relation)});
      }
    }
  }

  std::vector<std::pair<std::uint64_t, std::uint64_t>> scalarDependences;
  llvm::DenseMap<mlir::Operation *, std::uint64_t> statementOrdinals;
  for (auto [ordinal, statement] : llvm::enumerate(statements))
    statementOrdinals.try_emplace(statement.projected, ordinal);
  for (auto [destination, statement] : llvm::enumerate(statements)) {
    for (mlir::Value operand : statement.projected->getOperands()) {
      auto resultValue = llvm::dyn_cast<mlir::OpResult>(operand);
      if (!resultValue)
        continue;
      auto source = statementOrdinals.find(resultValue.getOwner());
      if (source == statementOrdinals.end() || source->second == destination)
        continue;
      const std::pair<std::uint64_t, std::uint64_t> edge{source->second,
                                                         destination};
      if (llvm::is_contained(scalarDependences, edge))
        continue;
      scalarDependences.push_back(edge);
      result.dependences.push_back(
          {StructuredPolyhedralDependenceKind::ScalarSsa, edge.first,
           edge.second, std::nullopt});
    }
  }

  std::vector<detail::PolyhedralStatementDomain> providerStatements;
  providerStatements.reserve(statements.size());
  for (auto [ordinal, statement] : llvm::enumerate(statements))
    providerStatements.push_back({ordinal, &statement.domain});
  std::vector<detail::PolyhedralDependenceRelation> providerDependences;
  providerDependences.reserve(memoryDependences.size() +
                              scalarDependences.size());
  for (const MemoryDependence &dependence : memoryDependences)
    providerDependences.push_back(
        {dependence.source, dependence.destination,
         statements[dependence.source].domain.getNumDimVars(),
         statements[dependence.destination].domain.getNumDimVars(),
         &dependence.relation});
  for (const auto &[source, destination] : scalarDependences)
    providerDependences.push_back(
        {source, destination, statements[source].domain.getNumDimVars(),
         statements[destination].domain.getNumDimVars(), nullptr});
  auto providerOutcome =
      detail::computePinnedIslSchedule(providerStatements, providerDependences);
  if (!providerOutcome)
    return providerOutcome.takeError();
  if (auto *providerRefusal =
          std::get_if<detail::PolyhedralScheduleProviderRefusalKind>(
              &*providerOutcome)) {
    StructuredScopRefusalKind refusal =
        StructuredScopRefusalKind::ProviderScheduleNotEstablished;
    if (*providerRefusal ==
        detail::PolyhedralScheduleProviderRefusalKind::DomainNotAdmitted)
      refusal = StructuredScopRefusalKind::ProviderDomainNotAdmitted;
    else if (*providerRefusal == detail::PolyhedralScheduleProviderRefusalKind::
                                     OperationBudgetExhausted)
      refusal = StructuredScopRefusalKind::ProviderScheduleBudgetExhausted;
    return refusePolyhedral(loopReference, refusal,
                            result.dependenceQueryCount);
  }
  detail::PolyhedralScheduleProviderView &providerSchedule =
      std::get<detail::PolyhedralScheduleProviderView>(*providerOutcome);
  if (providerSchedule.statementSchedules.size() != statements.size())
    return invalid("Polly/ISL schedule changed statement cardinality");
  if (providerSchedule.parameters.size() != providerSchedule.parameterCount)
    return invalid("Polly/ISL schedule changed parameter cardinality");
  result.parameters.reserve(providerSchedule.parameters.size());
  for (mlir::Value parameter : providerSchedule.parameters) {
    if (valueDefinedInside(parameter, projectedLoop->getOperation()))
      return refusePolyhedral(
          loopReference,
          StructuredScopRefusalKind::PolyhedralMaterializationUnavailable,
          result.dependenceQueryCount);
    auto reference = valueReference(parameter, projectedValueReferences);
    if (!reference)
      return reference.takeError();
    result.parameters.push_back(*reference);
  }
  result.schedule = {StructuredPolyhedralProviderKind::PinnedPollyIsl,
                     providerSchedule.form,
                     providerSchedule.parameterCount,
                     providerDependences.size(),
                     providerSchedule.scheduleBandCount,
                     providerSchedule.scheduleDimensionCount,
                     providerSchedule.coincidentDimensionCount,
                     std::move(providerSchedule.statementSchedules)};
  return StructuredPolyhedralScopAnalysisOutcome(std::move(result));
}

} // namespace loom::frontend
