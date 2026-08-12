#include "DataflowRewriteInternal.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/OperationSchema.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace dataflow::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "dataflow_cardinality_rewrite_invalid: " +
                                     message);
}

enum class Shell {
  ScalarComputeThenParallelize,
  SerializesThenScalarCompute,
  ParallelizesThenVectorCompute,
  VectorComputeThenSerialize,
};

struct Match final {
  Shell shell;
  mlir::Operation *compute;
  OperationSchemaId schema;
  llvm::SmallVector<mlir::Operation *, 4> adapters;
};

bool onlyUseIs(mlir::Value value, mlir::Operation *user) {
  return value.hasOneUse() && *value.getUsers().begin() == user;
}

std::optional<OperationSchemaId> elementwiseSchema(mlir::Operation *compute) {
  std::optional<OperationSchemaId> schema = operationSchemaOf(compute);
  if (!schema || actorKind(*schema) != CanonicalDataflowActorKind::Compute ||
      !supportsElementwiseVectorDecomposition(*schema) ||
      compute->getNumOperands() == 0 || compute->getNumResults() != 1 ||
      compute->getNumRegions() != 0 || compute->getNumSuccessors() != 0 ||
      !mlir::isPure(compute))
    return std::nullopt;
  return schema;
}

bool isScalarPayload(mlir::Type type) {
  return type.isIntOrIndexOrFloat() && !type.isIndex();
}

std::optional<mlir::VectorType> commonVectorShape(mlir::Operation *compute) {
  auto result =
      llvm::dyn_cast<mlir::VectorType>(compute->getResult(0).getType());
  if (!result || result.isScalable() || result.getRank() != 1)
    return std::nullopt;
  for (mlir::Value operand : compute->getOperands()) {
    auto type = llvm::dyn_cast<mlir::VectorType>(operand.getType());
    if (!type || type.isScalable() || type.getShape() != result.getShape())
      return std::nullopt;
  }
  return result;
}

bool alternativeSchemaIsRegistered(mlir::Operation *compute,
                                   OperationSchemaId schema,
                                   mlir::TypeRange operandTypes,
                                   mlir::Type resultType) {
  mlir::Operation *probe = compute->clone();
  for (auto [result, type] :
       llvm::zip(probe->getResults(), mlir::TypeRange{resultType}))
    result.setType(type);
  mlir::Block scratch;
  llvm::SmallVector<mlir::Value, 4> operands;
  for (mlir::Type type : operandTypes)
    operands.push_back(scratch.addArgument(type, compute->getLoc()));
  probe->setOperands(operands);
  llvm::Error canonical = canonicalizeRegisteredActorInstance(schema, probe);
  const bool valid = !canonical && operationSchemaOf(probe) == schema;
  if (canonical)
    llvm::consumeError(std::move(canonical));
  probe->erase();
  return valid;
}

std::optional<Match> matchScalarThenParallelize(mlir::Operation *compute,
                                                OperationSchemaId schema) {
  if (!llvm::all_of(compute->getOperandTypes(), isScalarPayload) ||
      !isScalarPayload(compute->getResult(0).getType()))
    return std::nullopt;
  if (!compute->getResult(0).hasOneUse())
    return std::nullopt;
  auto adapter =
      llvm::dyn_cast<ParallelizeOp>(*compute->getResult(0).getUsers().begin());
  if (!adapter || adapter.getData() != compute->getResult(0))
    return std::nullopt;
  auto resultType =
      llvm::dyn_cast<mlir::VectorType>(adapter.getVector().getType());
  if (!resultType || resultType.isScalable() || resultType.getRank() != 1 ||
      resultType.getElementType() != compute->getResult(0).getType())
    return std::nullopt;
  llvm::SmallVector<mlir::Type, 4> operandTypes;
  for (mlir::Type type : compute->getOperandTypes())
    operandTypes.push_back(mlir::VectorType::get(resultType.getShape(), type));
  if (!alternativeSchemaIsRegistered(compute, schema, operandTypes, resultType))
    return std::nullopt;
  return Match{Shell::ScalarComputeThenParallelize,
               compute,
               schema,
               {adapter.getOperation()}};
}

std::optional<Match> matchSerializesThenScalar(mlir::Operation *compute,
                                               OperationSchemaId schema) {
  if (!llvm::all_of(compute->getOperandTypes(), isScalarPayload) ||
      !isScalarPayload(compute->getResult(0).getType()))
    return std::nullopt;
  llvm::SmallVector<mlir::Operation *, 4> adapters;
  mlir::Value commonMask;
  mlir::Value commonPhase;
  llvm::SmallVector<mlir::Type, 4> operandTypes;
  llvm::ArrayRef<std::int64_t> shape;
  for (auto [ordinal, operand] : llvm::enumerate(compute->getOperands())) {
    auto result = llvm::dyn_cast<mlir::OpResult>(operand);
    auto adapter =
        result ? llvm::dyn_cast<SerializeOp>(result.getOwner()) : SerializeOp{};
    if (!adapter || result.getResultNumber() != 0 ||
        !onlyUseIs(adapter.getData(), compute))
      return std::nullopt;
    auto vectorType =
        llvm::dyn_cast<mlir::VectorType>(adapter.getVector().getType());
    if (!vectorType || vectorType.isScalable() || vectorType.getRank() != 1 ||
        vectorType.getElementType() != operand.getType())
      return std::nullopt;
    if (ordinal == 0) {
      commonMask = adapter.getMask();
      commonPhase = adapter.getGroupPhase();
      shape = vectorType.getShape();
    } else if (adapter.getMask() != commonMask ||
               adapter.getGroupPhase() != commonPhase ||
               vectorType.getShape() != shape ||
               !adapter.getScalarPhase().use_empty()) {
      return std::nullopt;
    }
    adapters.push_back(adapter.getOperation());
    operandTypes.push_back(vectorType);
  }
  mlir::VectorType resultType =
      mlir::VectorType::get(shape, compute->getResult(0).getType());
  if (!alternativeSchemaIsRegistered(compute, schema, operandTypes, resultType))
    return std::nullopt;
  return Match{Shell::SerializesThenScalarCompute, compute, schema,
               std::move(adapters)};
}

std::optional<Match> matchParallelizesThenVector(mlir::Operation *compute,
                                                 OperationSchemaId schema) {
  std::optional<mlir::VectorType> resultType = commonVectorShape(compute);
  if (!resultType)
    return std::nullopt;
  llvm::SmallVector<mlir::Operation *, 4> adapters;
  llvm::SmallVector<mlir::Type, 4> scalarTypes;
  mlir::Value commonPhase;
  for (auto [ordinal, operand] : llvm::enumerate(compute->getOperands())) {
    auto result = llvm::dyn_cast<mlir::OpResult>(operand);
    auto adapter = result ? llvm::dyn_cast<ParallelizeOp>(result.getOwner())
                          : ParallelizeOp{};
    if (!adapter || result.getResultNumber() != 0 ||
        !onlyUseIs(adapter.getVector(), compute) ||
        adapter.getVector().getType() != operand.getType())
      return std::nullopt;
    if (ordinal == 0) {
      commonPhase = adapter.getScalarPhase();
    } else if (adapter.getScalarPhase() != commonPhase ||
               !adapter.getMask().use_empty() ||
               !adapter.getGroupPhase().use_empty()) {
      return std::nullopt;
    }
    adapters.push_back(adapter.getOperation());
    scalarTypes.push_back(adapter.getData().getType());
  }
  if (!alternativeSchemaIsRegistered(compute, schema, scalarTypes,
                                     resultType->getElementType()))
    return std::nullopt;
  return Match{Shell::ParallelizesThenVectorCompute, compute, schema,
               std::move(adapters)};
}

std::optional<Match> matchVectorThenSerialize(mlir::Operation *compute,
                                              OperationSchemaId schema) {
  std::optional<mlir::VectorType> resultType = commonVectorShape(compute);
  if (!resultType || !compute->getResult(0).hasOneUse())
    return std::nullopt;
  auto adapter =
      llvm::dyn_cast<SerializeOp>(*compute->getResult(0).getUsers().begin());
  if (!adapter || adapter.getVector() != compute->getResult(0) ||
      adapter.getData().getType() != resultType->getElementType())
    return std::nullopt;
  llvm::SmallVector<mlir::Type, 4> scalarTypes;
  for (mlir::Type type : compute->getOperandTypes())
    scalarTypes.push_back(llvm::cast<mlir::VectorType>(type).getElementType());
  if (!alternativeSchemaIsRegistered(compute, schema, scalarTypes,
                                     adapter.getData().getType()))
    return std::nullopt;
  return Match{Shell::VectorComputeThenSerialize,
               compute,
               schema,
               {adapter.getOperation()}};
}

std::vector<Match> matchesFor(mlir::Operation *compute,
                              CardinalityCommuteDirection direction) {
  std::vector<Match> matches;
  std::optional<OperationSchemaId> schema = elementwiseSchema(compute);
  if (!schema)
    return matches;
  const auto append = [&](std::optional<Match> match) {
    if (match)
      matches.push_back(std::move(*match));
  };
  if (direction == CardinalityCommuteDirection::MoveInside) {
    append(matchScalarThenParallelize(compute, *schema));
    append(matchSerializesThenScalar(compute, *schema));
  } else if (direction == CardinalityCommuteDirection::MoveOutside) {
    append(matchParallelizesThenVector(compute, *schema));
    append(matchVectorThenSerialize(compute, *schema));
  }
  return matches;
}

llvm::Expected<std::vector<ActorId>>
adapterIds(const Match &match,
           const llvm::DenseMap<mlir::Operation *, ActorId> &ids) {
  std::vector<ActorId> result;
  for (mlir::Operation *adapter : match.adapters) {
    auto found = ids.find(adapter);
    if (found == ids.end())
      return invalid("adapter is outside the canonical actor inventory");
    result.push_back(found->second);
  }
  llvm::sort(result, [](ActorId lhs, ActorId rhs) {
    return lhs.value() < rhs.value();
  });
  return result;
}

mlir::Operation *cloneCompute(mlir::OpBuilder &builder, const Match &match,
                              mlir::ValueRange operands,
                              mlir::Type resultType) {
  mlir::Operation *clone = match.compute->clone();
  clone->removeAttr(kEntityIdAttrName);
  clone->setOperands(operands);
  clone->getResult(0).setType(resultType);
  builder.insert(clone);
  if (llvm::Error error =
          canonicalizeRegisteredActorInstance(match.schema, clone)) {
    llvm::consumeError(std::move(error));
    clone->erase();
    return nullptr;
  }
  return clone;
}

ParallelizeOp createParallelize(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value data, mlir::Value phase,
                                llvm::ArrayRef<std::int64_t> shape) {
  auto vectorType = mlir::VectorType::get(shape, data.getType());
  auto maskType = mlir::VectorType::get(shape, builder.getI1Type());
  return ParallelizeOp::create(
      builder, loc, mlir::TypeRange{vectorType, maskType, builder.getI1Type()},
      mlir::ValueRange{data, phase});
}

SerializeOp createSerialize(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value vector, mlir::Value mask,
                            mlir::Value phase) {
  auto vectorType = llvm::cast<mlir::VectorType>(vector.getType());
  return SerializeOp::create(
      builder, loc,
      mlir::TypeRange{vectorType.getElementType(), builder.getI1Type()},
      mlir::ValueRange{vector, mask, phase});
}

llvm::Error applyMatch(const Match &match) {
  mlir::OpBuilder builder(match.compute);
  if (match.shell == Shell::ScalarComputeThenParallelize) {
    auto resultAdapter = llvm::cast<ParallelizeOp>(match.adapters.front());
    auto resultType =
        llvm::cast<mlir::VectorType>(resultAdapter.getVector().getType());
    llvm::SmallVector<ParallelizeOp, 4> inputs;
    for (mlir::Value operand : match.compute->getOperands())
      inputs.push_back(createParallelize(
          builder, match.compute->getLoc(), operand,
          resultAdapter.getScalarPhase(), resultType.getShape()));
    llvm::SmallVector<mlir::Value, 4> vectors;
    for (ParallelizeOp input : inputs)
      vectors.push_back(input.getVector());
    mlir::Operation *compute =
        cloneCompute(builder, match, vectors, resultType);
    if (!compute)
      return invalid("cannot construct registered vector Compute");
    resultAdapter.getVector().replaceAllUsesWith(compute->getResult(0));
    resultAdapter.getMask().replaceAllUsesWith(inputs.front().getMask());
    resultAdapter.getGroupPhase().replaceAllUsesWith(
        inputs.front().getGroupPhase());
  } else if (match.shell == Shell::SerializesThenScalarCompute) {
    llvm::SmallVector<SerializeOp, 4> inputs;
    llvm::SmallVector<mlir::Value, 4> vectors;
    for (mlir::Operation *adapter : match.adapters) {
      auto serialize = llvm::cast<SerializeOp>(adapter);
      inputs.push_back(serialize);
      vectors.push_back(serialize.getVector());
    }
    auto representative = inputs.front();
    auto resultType = mlir::VectorType::get(
        llvm::cast<mlir::VectorType>(vectors.front().getType()).getShape(),
        match.compute->getResult(0).getType());
    mlir::Operation *compute =
        cloneCompute(builder, match, vectors, resultType);
    if (!compute)
      return invalid("cannot construct registered vector Compute");
    auto result = createSerialize(
        builder, match.compute->getLoc(), compute->getResult(0),
        representative.getMask(), representative.getGroupPhase());
    match.compute->getResult(0).replaceAllUsesWith(result.getData());
    representative.getScalarPhase().replaceAllUsesWith(result.getScalarPhase());
  } else if (match.shell == Shell::ParallelizesThenVectorCompute) {
    llvm::SmallVector<ParallelizeOp, 4> inputs;
    llvm::SmallVector<mlir::Value, 4> scalars;
    for (mlir::Operation *adapter : match.adapters) {
      auto parallelize = llvm::cast<ParallelizeOp>(adapter);
      inputs.push_back(parallelize);
      scalars.push_back(parallelize.getData());
    }
    auto representative = inputs.front();
    mlir::Operation *compute = cloneCompute(
        builder, match, scalars,
        llvm::cast<mlir::VectorType>(match.compute->getResult(0).getType())
            .getElementType());
    if (!compute)
      return invalid("cannot construct registered scalar Compute");
    auto result = createParallelize(
        builder, match.compute->getLoc(), compute->getResult(0),
        representative.getScalarPhase(),
        llvm::cast<mlir::VectorType>(match.compute->getResult(0).getType())
            .getShape());
    match.compute->getResult(0).replaceAllUsesWith(result.getVector());
    representative.getMask().replaceAllUsesWith(result.getMask());
    representative.getGroupPhase().replaceAllUsesWith(result.getGroupPhase());
  } else {
    auto resultAdapter = llvm::cast<SerializeOp>(match.adapters.front());
    llvm::SmallVector<SerializeOp, 4> inputs;
    llvm::SmallVector<mlir::Value, 4> scalars;
    for (mlir::Value operand : match.compute->getOperands()) {
      auto serialize = createSerialize(builder, match.compute->getLoc(),
                                       operand, resultAdapter.getMask(),
                                       resultAdapter.getGroupPhase());
      inputs.push_back(serialize);
      scalars.push_back(serialize.getData());
    }
    mlir::Operation *compute = cloneCompute(builder, match, scalars,
                                            resultAdapter.getData().getType());
    if (!compute)
      return invalid("cannot construct registered scalar Compute");
    resultAdapter.getData().replaceAllUsesWith(compute->getResult(0));
    resultAdapter.getScalarPhase().replaceAllUsesWith(
        inputs.front().getScalarPhase());
  }

  const bool resultSide = match.shell == Shell::ScalarComputeThenParallelize ||
                          match.shell == Shell::VectorComputeThenSerialize;
  if (resultSide)
    for (mlir::Operation *adapter : match.adapters)
      adapter->erase();
  match.compute->erase();
  if (!resultSide)
    for (mlir::Operation *adapter : match.adapters)
      adapter->erase();
  return llvm::Error::success();
}

} // namespace

llvm::Expected<std::vector<DataflowRewriteDecision>>
enumerateCardinalityCommuteDecisions(const CanonicalDataflowArtifact &parent) {
  auto view = parent.view();
  if (!view)
    return view.takeError();
  llvm::DenseMap<mlir::Operation *, ActorId> ids;
  for (const CanonicalActorView &actor : view->actors())
    ids.try_emplace(actor.op, actor.ref.entity);

  std::vector<DataflowRewriteDecision> decisions;
  for (const CanonicalActorView &actor : view->actors()) {
    for (CardinalityCommuteDirection direction :
         {CardinalityCommuteDirection::MoveInside,
          CardinalityCommuteDirection::MoveOutside}) {
      for (const Match &match : matchesFor(actor.op, direction)) {
        auto adapters = adapterIds(match, ids);
        if (!adapters)
          return adapters.takeError();
        decisions.emplace_back(ElementwiseCardinalityCommuteRewrite{
            actor.ref.entity, std::move(*adapters), direction});
      }
    }
  }
  llvm::sort(decisions, dataflowRewriteDecisionLess);
  return decisions;
}

llvm::Expected<std::optional<MaterializedDataflowRewriteProjection>>
materializeCardinalityCommuteRewriteProjection(
    const CanonicalDataflowArtifact &parent,
    const ElementwiseCardinalityCommuteRewrite &decision,
    llvm::ArrayRef<StaticGraphLaunchRef> trackedStaticGraphLaunches) {
  auto decisions = enumerateCardinalityCommuteDecisions(parent);
  if (!decisions)
    return decisions.takeError();
  if (!llvm::is_contained(*decisions, DataflowRewriteDecision{decision}))
    return invalid("decision is not a complete legal parent shell");

  auto view = parent.view();
  if (!view)
    return view.takeError();
  auto resolved = view->resolve(ActorRef{parent.identity(), decision.compute});
  if (!resolved)
    return resolved.takeError();
  mlir::IRMapping mapping;
  mlir::OwningOpRef<mlir::ModuleOp> candidate(
      mlir::cast<mlir::ModuleOp>(parent.module()->clone(mapping)));
  mlir::Operation *compute = mapping.lookupOrNull(resolved->op);
  if (!compute)
    return invalid("compute was not cloned into the candidate");
  std::vector<Match> matches = matchesFor(compute, decision.direction);
  llvm::SmallPtrSet<mlir::Operation *, 4> expectedAdapters;
  for (ActorId id : decision.adapters) {
    auto adapter = view->resolve(ActorRef{parent.identity(), id});
    if (!adapter)
      return adapter.takeError();
    mlir::Operation *cloned = mapping.lookupOrNull(adapter->op);
    if (!cloned)
      return invalid("adapter was not cloned into the candidate");
    expectedAdapters.insert(cloned);
  }
  auto selected = llvm::find_if(matches, [&](const Match &match) {
    return match.adapters.size() == expectedAdapters.size() &&
           llvm::all_of(match.adapters, [&](mlir::Operation *adapter) {
             return expectedAdapters.contains(adapter);
           });
  });
  if (selected == matches.end())
    return invalid("cloned shell is ambiguous or absent");
  if (llvm::Error error = applyMatch(*selected))
    return std::move(error);

  return finalizeDataflowRewriteCandidate(parent, candidate.get(), mapping,
                                          trackedStaticGraphLaunches);
}

llvm::Expected<std::optional<CanonicalDataflowArtifact>>
materializeCardinalityCommuteRewrite(
    const CanonicalDataflowArtifact &parent,
    const ElementwiseCardinalityCommuteRewrite &decision) {
  auto projected =
      materializeCardinalityCommuteRewriteProjection(parent, decision, {});
  if (!projected)
    return projected.takeError();
  if (!*projected)
    return std::optional<CanonicalDataflowArtifact>{};
  return std::optional<CanonicalDataflowArtifact>(
      std::move((*projected)->artifact));
}

} // namespace dataflow::detail
