#include "StructuredScheduleCapacity.h"

#include "StructuredScheduleInternal.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Frontend/Compilation/StructuredSpecialMathAccuracy.h"
#include "Frontend/IR/LoomDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace loom::frontend::detail {
namespace {

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
  if (!hasUnresolvedStructuredSpecialMathAccuracy(operation)) {
    auto key = dataflow::projectRegisteredActorSchemaProjectionBytes(operation);
    if (!key)
      return key.takeError();
    return AggregateUnrollActorProjection{std::move(*key), std::nullopt};
  }

  auto projections = projectStructuredSpecialMathAccuracyDomain(operation);
  if (!projections)
    return projections.takeError();
  if (projections->empty())
    return detail::invalidStructuredSchedule("unresolved special-math domain is empty");
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

} // namespace

bool isRemovableAddressSupport(mlir::Operation *operation) {
  auto address = llvm::dyn_cast<mlir::LLVM::GEPOp>(operation);
  if (!address)
    return false;
  llvm::SmallVector<mlir::LLVM::GEPOp, 4> pending{address};
  llvm::SmallPtrSet<mlir::Operation *, 4> visited;
  while (!pending.empty()) {
    mlir::LLVM::GEPOp current = pending.pop_back_val();
    if (!visited.insert(current.getOperation()).second)
      continue;
    if (current->use_empty())
      return false;
    for (mlir::Operation *user : current->getUsers()) {
      if (auto chained = llvm::dyn_cast<mlir::LLVM::GEPOp>(user)) {
        if (chained.getBase() != current.getResult())
          return false;
        pending.push_back(chained);
        continue;
      }
      if (!llvm::isa_and_nonnull<mlir::UnitAttr>(
              user->getAttr(loom::rootRelativeAddressAttrName)))
        return false;
      if (auto load = llvm::dyn_cast<mlir::LLVM::LoadOp>(user)) {
        if (load.getVolatile_() ||
            load.getOrdering() != mlir::LLVM::AtomicOrdering::not_atomic ||
            load.getAddr() != current.getResult())
          return false;
        continue;
      }
      if (auto store = llvm::dyn_cast<mlir::LLVM::StoreOp>(user)) {
        if (store.getVolatile_() ||
            store.getOrdering() != mlir::LLVM::AtomicOrdering::not_atomic ||
            store.getAddr() != current.getResult())
          return false;
        continue;
      }
      return false;
    }
  }
  return true;
}

llvm::Expected<std::uint64_t>
admittingStructuredActorResources(mlir::Operation *operation,
                                  const FabricCapabilityIndex &fabric) {
  const std::optional<dataflow::OperationSchemaId> schema =
      dataflow::operationSchemaOf(operation);
  if (!schema)
    return detail::invalidStructuredSchedule("structured actor has no operation schema");
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

llvm::Expected<AggregateReplicationBound>
aggregateUnrollCapacity(mlir::scf::ForOp loop,
                        const FabricCapabilityIndex &fabric) {
  std::map<std::vector<std::uint8_t>, ActorMultiplicity> actors;
  llvm::Error projectionError = llvm::Error::success();
  loop.getRegion().walk([&](mlir::Operation *operation) {
    if (projectionError || !dataflow::operationSchemaOf(operation))
      return mlir::WalkResult::advance();
    // Removable address support is not replicated capacity: the replicated
    // body folds it into the accesses it serves, exactly as the
    // materialization gate does.
    if (isRemovableAddressSupport(operation))
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
      projectionError = detail::invalidStructuredSchedule("actor-equivalent capacity bounds disagree");
      return mlir::WalkResult::interrupt();
    }
    const std::optional<std::uint64_t> next =
        llvm::checkedAddUnsigned(multiplicity.count, std::uint64_t{1});
    if (!next) {
      projectionError = detail::invalidStructuredSchedule("actor multiplicity overflow");
      return mlir::WalkResult::interrupt();
    }
    multiplicity.count = *next;
    return mlir::WalkResult::advance();
  });
  if (projectionError)
    return std::move(projectionError);
  if (actors.empty())
    return AggregateReplicationBound{};

  AggregateReplicationBound bound;
  bound.factor = std::numeric_limits<std::uint64_t>::max();
  for (const auto &entry : actors) {
    mlir::Operation *actor = entry.second.representative;
    auto kind = dataflow::classifyCanonicalDataflowActor(actor);
    if (!kind)
      return detail::invalidStructuredSchedule("registered actor lost its canonical kind");
    std::uint64_t resources = 0;
    if (entry.second.resourceUpperBound) {
      resources = *entry.second.resourceUpperBound;
    } else {
      // The same admission projection the materialization gate applies, so a
      // constant or a memory actor counts the resources that really admit it.
      llvm::Expected<std::uint64_t> admitted =
          admittingStructuredActorResources(actor, fabric);
      if (!admitted)
        return admitted.takeError();
      resources = *admitted;
    }
    const std::uint64_t factor = resources / entry.second.count;
    if (factor < bound.factor)
      bound = {factor, actor->getName().getStringRef(), entry.second.count,
               resources};
  }
  return bound;
}

} // namespace loom::frontend::detail
