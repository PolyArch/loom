#include "Dataflow/Transforms/DataflowRewrite.h"

#include "DataflowRewriteInternal.h"

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <utility>
#include <variant>

namespace dataflow {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "dataflow_vector_rewrite_invalid: " + message);
}

struct ElementwiseVectorActor final {
  mlir::Operation *operation;
  llvm::SmallVector<mlir::VectorType, 4> operandTypes;
  mlir::VectorType resultType;
  std::uint64_t elementCount;
  OperationSchemaId schema;
};

llvm::Expected<std::optional<ElementwiseVectorActor>>
analyzeElementwiseVectorActor(const CanonicalDataflowArtifact &parent,
                              ActorRef actor) {
  auto view = parent.view();
  if (!view)
    return view.takeError();
  auto resolved = view->resolve(actor);
  if (!resolved)
    return resolved.takeError();
  mlir::Operation *operation = resolved->op;
  std::optional<OperationSchemaId> schema = operationSchemaOf(operation);
  if (resolved->kind != CanonicalDataflowActorKind::Compute || !schema ||
      actorKind(*schema) != CanonicalDataflowActorKind::Compute ||
      !supportsElementwiseVectorDecomposition(*schema) ||
      operation->getNumOperands() == 0 || operation->getNumResults() != 1 ||
      operation->getNumRegions() != 0 || operation->getNumSuccessors() != 0)
    return std::optional<ElementwiseVectorActor>{};

  auto resultType =
      llvm::dyn_cast<mlir::VectorType>(operation->getResult(0).getType());
  if (!resultType || resultType.isScalable() || resultType.getRank() == 0)
    return std::optional<ElementwiseVectorActor>{};

  llvm::SmallVector<mlir::VectorType, 4> operandTypes;
  operandTypes.reserve(operation->getNumOperands());
  for (mlir::Value operand : operation->getOperands()) {
    auto type = llvm::dyn_cast<mlir::VectorType>(operand.getType());
    if (!type || type.isScalable() || type.getShape() != resultType.getShape())
      return std::optional<ElementwiseVectorActor>{};
    operandTypes.push_back(type);
  }

  std::uint64_t elementCount = 1;
  for (std::int64_t dimension : resultType.getShape()) {
    if (dimension <= 0)
      return invalid("fixed vector has a non-positive dimension");
    auto product = llvm::checkedMulUnsigned(
        elementCount, static_cast<std::uint64_t>(dimension));
    if (!product)
      return invalid("fixed vector element count overflows u64");
    elementCount = *product;
  }
  return std::optional<ElementwiseVectorActor>{ElementwiseVectorActor{
      operation, std::move(operandTypes), resultType, elementCount, *schema}};
}

mlir::VectorType withLeadingBlocks(mlir::VectorType type,
                                   std::int64_t leadingBlocks) {
  llvm::SmallVector<std::int64_t, 4> shape(type.getShape());
  shape.front() = leadingBlocks;
  return mlir::VectorType::get(shape, type.getElementType());
}

llvm::SmallVector<mlir::Value, 4> jointInputs(mlir::OpBuilder &builder,
                                              mlir::Operation *operation) {
  llvm::SmallVector<mlir::Value, 4> inputs(operation->getOperands());
  if (inputs.size() < 2)
    return inputs;
  llvm::SmallVector<mlir::Type, 4> types;
  types.reserve(inputs.size());
  for (mlir::Value input : inputs)
    types.push_back(input.getType());
  auto sync = SyncOp::create(builder, operation->getLoc(), types, inputs);
  return llvm::SmallVector<mlir::Value, 4>(sync.getOutputs());
}

llvm::Expected<mlir::Value> cloneElementwiseActor(mlir::OpBuilder &builder,
                                                  mlir::Operation *operation,
                                                  OperationSchemaId schema,
                                                  mlir::ValueRange operands,
                                                  mlir::Type resultType) {
  mlir::Operation *clone = operation->clone();
  clone->removeAttr(kEntityIdAttrName);
  clone->setOperands(operands);
  clone->getResult(0).setType(resultType);
  builder.insert(clone);
  if (llvm::Error error = canonicalizeRegisteredActorInstance(schema, clone)) {
    clone->erase();
    return std::move(error);
  }
  return clone->getResult(0);
}

mlir::Value extractLeadingChunk(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value source, std::int64_t firstBlock,
                                std::int64_t blockCount) {
  auto sourceType = llvm::cast<mlir::VectorType>(source.getType());
  mlir::VectorType resultType = withLeadingBlocks(sourceType, blockCount);
  llvm::SmallVector<std::int64_t, 8> mask;
  mask.reserve(blockCount);
  for (std::int64_t offset = 0; offset != blockCount; ++offset)
    mask.push_back(firstBlock + offset);
  return mlir::vector::ShuffleOp::create(builder, loc, resultType, source,
                                         source, mask)
      .getVector();
}

mlir::Value appendLeadingChunk(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value prefix, mlir::Value suffix) {
  auto prefixType = llvm::cast<mlir::VectorType>(prefix.getType());
  auto suffixType = llvm::cast<mlir::VectorType>(suffix.getType());
  const std::int64_t total =
      prefixType.getDimSize(0) + suffixType.getDimSize(0);
  mlir::VectorType resultType = withLeadingBlocks(prefixType, total);
  llvm::SmallVector<std::int64_t, 8> mask;
  mask.reserve(total);
  for (std::int64_t ordinal = 0; ordinal != total; ++ordinal)
    mask.push_back(ordinal);
  return mlir::vector::ShuffleOp::create(builder, loc, resultType, prefix,
                                         suffix, mask)
      .getVector();
}

llvm::Error applyChunkRewrite(mlir::Operation *operation,
                              OperationSchemaId schema,
                              std::int64_t leadingBlocksPerChunk) {
  auto resultType =
      llvm::cast<mlir::VectorType>(operation->getResult(0).getType());
  const std::int64_t leadingBlocks = resultType.getDimSize(0);
  if (leadingBlocksPerChunk == 0 || leadingBlocksPerChunk >= leadingBlocks ||
      leadingBlocks % leadingBlocksPerChunk != 0)
    return invalid("chunk size is not a proper leading-dimension divisor");

  mlir::OpBuilder builder(operation);
  llvm::SmallVector<mlir::Value, 4> inputs = jointInputs(builder, operation);
  const std::int64_t chunkCount = leadingBlocks / leadingBlocksPerChunk;
  llvm::SmallVector<mlir::Value, 8> chunkResults;
  chunkResults.reserve(chunkCount);
  for (std::int64_t chunk = 0; chunk != chunkCount; ++chunk) {
    llvm::SmallVector<mlir::Value, 4> chunkInputs;
    chunkInputs.reserve(inputs.size());
    for (mlir::Value input : inputs)
      chunkInputs.push_back(extractLeadingChunk(
          builder, operation->getLoc(), input, chunk * leadingBlocksPerChunk,
          leadingBlocksPerChunk));
    auto result = cloneElementwiseActor(
        builder, operation, schema, chunkInputs,
        withLeadingBlocks(resultType, leadingBlocksPerChunk));
    if (!result)
      return result.takeError();
    chunkResults.push_back(*result);
  }

  mlir::Value result = chunkResults.front();
  for (mlir::Value chunk : llvm::drop_begin(chunkResults))
    result = appendLeadingChunk(builder, operation->getLoc(), result, chunk);
  operation->getResult(0).replaceAllUsesWith(result);
  operation->erase();
  return llvm::Error::success();
}

llvm::SmallVector<std::int64_t, 4> rowMajorPosition(mlir::VectorType type,
                                                    std::uint64_t linear) {
  llvm::SmallVector<std::int64_t, 4> position(type.getRank());
  for (std::int64_t dimension = type.getRank(); dimension-- > 0;) {
    const std::uint64_t extent =
        static_cast<std::uint64_t>(type.getDimSize(dimension));
    position[dimension] = static_cast<std::int64_t>(linear % extent);
    linear /= extent;
  }
  return position;
}

llvm::Error applyScalarizeRewrite(mlir::Operation *operation,
                                  OperationSchemaId schema,
                                  std::uint64_t elementCount) {
  auto resultType =
      llvm::cast<mlir::VectorType>(operation->getResult(0).getType());
  mlir::OpBuilder builder(operation);
  llvm::SmallVector<mlir::Value, 4> inputs = jointInputs(builder, operation);

  auto base = llvm::find_if(
      inputs, [&](mlir::Value input) { return input.getType() == resultType; });
  if (base == inputs.end())
    return invalid("scalarization has no exact result-typed base operand");

  llvm::SmallVector<mlir::Value, 16> scalarResults;
  scalarResults.reserve(elementCount);
  for (std::uint64_t linear = 0; linear != elementCount; ++linear) {
    llvm::SmallVector<std::int64_t, 4> position =
        rowMajorPosition(resultType, linear);
    llvm::SmallVector<mlir::Value, 4> scalarInputs;
    scalarInputs.reserve(inputs.size());
    for (mlir::Value input : inputs)
      scalarInputs.push_back(mlir::vector::ExtractOp::create(
          builder, operation->getLoc(), input, position));
    auto scalar = cloneElementwiseActor(
        builder, operation, schema, scalarInputs, resultType.getElementType());
    if (!scalar)
      return scalar.takeError();
    scalarResults.push_back(*scalar);
  }

  mlir::Value result = *base;
  for (std::uint64_t linear = 0; linear != elementCount; ++linear) {
    llvm::SmallVector<std::int64_t, 4> position =
        rowMajorPosition(resultType, linear);
    result = mlir::vector::InsertOp::create(
        builder, operation->getLoc(), scalarResults[linear], result, position);
  }
  operation->getResult(0).replaceAllUsesWith(result);
  operation->erase();
  return llvm::Error::success();
}

llvm::Expected<ElementwiseVectorActor>
validateElementwiseDecision(const CanonicalDataflowArtifact &parent,
                            const DataflowRewriteDecision &decision) {
  if (dataflowRewriteKind(decision) !=
      DataflowRewriteKind::ElementwiseVectorDecompose)
    return invalid("decision is not an elementwise vector decomposition");
  const ActorId actorId =
      std::holds_alternative<ElementwiseVectorChunkRewrite>(decision)
          ? std::get<ElementwiseVectorChunkRewrite>(decision).compute
          : std::get<ElementwiseVectorScalarizeRewrite>(decision).compute;
  const ActorRef actor{parent.identity(), actorId};
  auto analyzed = analyzeElementwiseVectorActor(parent, actor);
  if (!analyzed)
    return analyzed.takeError();
  if (!*analyzed)
    return invalid("decision actor is outside the decomposition domain");

  if (const auto *chunk =
          std::get_if<ElementwiseVectorChunkRewrite>(&decision)) {
    const std::int64_t leadingBlocks = (*analyzed)->resultType.getDimSize(0);
    if (chunk->leadingBlocksPerChunk >=
            static_cast<std::uint64_t>(leadingBlocks) ||
        static_cast<std::uint64_t>(leadingBlocks) %
                chunk->leadingBlocksPerChunk !=
            0)
      return invalid("chunk decision is not a proper leading-dimension "
                     "divisor");
  } else if (!llvm::is_contained((*analyzed)->operandTypes,
                                 (*analyzed)->resultType)) {
    return invalid("scalarization has no exact result-typed base operand");
  }
  return std::move(**analyzed);
}

std::vector<std::int64_t> properLeadingDivisors(std::int64_t leadingBlocks) {
  std::vector<std::int64_t> divisors;
  for (std::int64_t factor = 1; factor <= leadingBlocks / factor; ++factor) {
    if (leadingBlocks % factor != 0)
      continue;
    if (factor < leadingBlocks)
      divisors.push_back(factor);
    const std::int64_t paired = leadingBlocks / factor;
    if (paired != factor && paired < leadingBlocks)
      divisors.push_back(paired);
  }
  llvm::sort(divisors, std::greater<std::int64_t>());
  divisors.erase(std::unique(divisors.begin(), divisors.end()), divisors.end());
  return divisors;
}

} // namespace

llvm::Expected<std::vector<DataflowRewriteDecision>>
enumerateElementwiseVectorDecompositionDecisions(
    const CanonicalDataflowArtifact &parent, ActorRef actor) {
  auto analyzed = analyzeElementwiseVectorActor(parent, actor);
  if (!analyzed)
    return analyzed.takeError();
  std::vector<DataflowRewriteDecision> decisions;
  if (!*analyzed)
    return decisions;

  const std::int64_t leadingBlocks = (*analyzed)->resultType.getDimSize(0);
  for (std::int64_t chunk : properLeadingDivisors(leadingBlocks))
    decisions.emplace_back(ElementwiseVectorChunkRewrite{
        actor.entity, static_cast<std::uint64_t>(chunk)});

  if (llvm::is_contained((*analyzed)->operandTypes, (*analyzed)->resultType))
    decisions.emplace_back(ElementwiseVectorScalarizeRewrite{actor.entity});
  return decisions;
}

llvm::Expected<std::uint64_t>
dataflowRewriteExpansionCost(const CanonicalDataflowArtifact &parent,
                             const DataflowRewriteDecision &decision) {
  auto encoded = encodeDataflowRewriteDecision(decision);
  if (!encoded)
    return encoded.takeError();
  if (dataflowRewriteKind(decision) !=
      DataflowRewriteKind::ElementwiseVectorDecompose)
    return 1;
  auto analyzed = validateElementwiseDecision(parent, decision);
  if (!analyzed)
    return analyzed.takeError();
  if (const auto *chunk =
          std::get_if<ElementwiseVectorChunkRewrite>(&decision)) {
    const std::uint64_t leadingBlocks =
        static_cast<std::uint64_t>(analyzed->resultType.getDimSize(0));
    return leadingBlocks / chunk->leadingBlocksPerChunk;
  }
  return analyzed->elementCount;
}

llvm::Expected<std::optional<CanonicalDataflowArtifact>>
materializeDataflowRewrite(const CanonicalDataflowArtifact &parent,
                           const DataflowRewriteDecision &decision) {
  auto encoded = encodeDataflowRewriteDecision(decision);
  if (!encoded)
    return encoded.takeError();
  if (dataflowRewriteKind(decision) !=
      DataflowRewriteKind::ElementwiseVectorDecompose)
    return detail::materializeFixedDataflowRewrite(parent, decision);

  auto analyzed = validateElementwiseDecision(parent, decision);
  if (!analyzed)
    return analyzed.takeError();

  mlir::IRMapping mapping;
  mlir::OwningOpRef<mlir::ModuleOp> candidate(
      mlir::cast<mlir::ModuleOp>(parent.module()->clone(mapping)));
  mlir::Operation *operation = mapping.lookupOrNull(analyzed->operation);
  if (!operation)
    return invalid("selected actor was not cloned into the candidate");

  if (const auto *chunk =
          std::get_if<ElementwiseVectorChunkRewrite>(&decision)) {
    if (llvm::Error error = applyChunkRewrite(
            operation, analyzed->schema,
            static_cast<std::int64_t>(chunk->leadingBlocksPerChunk)))
      return std::move(error);
  } else if (llvm::Error error = applyScalarizeRewrite(
                 operation, analyzed->schema, analyzed->elementCount)) {
    return std::move(error);
  }

  auto finalized = finalizeCanonicalDataflow(candidate.get());
  if (!finalized)
    return finalized.takeError();
  if (finalized->identity() == parent.identity())
    return std::optional<CanonicalDataflowArtifact>{};
  return std::optional<CanonicalDataflowArtifact>(std::move(*finalized));
}

} // namespace dataflow
