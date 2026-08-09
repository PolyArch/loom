#include "Dataflow/IR/DataflowCanonicalArtifact.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/OperationSchema.h"

#include "llvm/ADT/STLExtras.h"

#include <cstddef>
#include <cstdint>
#include <numeric>

namespace dataflow {
namespace {

bool isMemoryCapability(mlir::Type type) {
  return DataflowDialect::isMemoryCapabilityType(type);
}

} // namespace

llvm::Expected<ActivityDefinedness>
CanonicalDataflowProgramView::activityDefinedness(
    const CanonicalGraphProducerEndpointRef &producer) const {
  if (llvm::Error error = validate(producer))
    return std::move(error);
  if (std::holds_alternative<GraphIngressTokenRef>(producer))
    return ActivityDefinedness::Unproven;
  const auto &result = std::get<ActorTokenResultRef>(producer);
  const std::size_t slot = slotOfId_[result.actor.entity.value()];
  return activityDefinednessByActorSlot_[slot][result.ordinal];
}

llvm::Error CanonicalDataflowProgramView::buildActivityDefinednessProjection() {
  activityDefinednessByActorSlot_.clear();
  activityDefinednessByActorSlot_.reserve(actors_.size());
  for (const CanonicalActorView &actor : actors_)
    activityDefinednessByActorSlot_.push_back(std::vector<ActivityDefinedness>(
        actor.op->getNumResults(), ActivityDefinedness::Unproven));

  auto operandsAreDefined =
      [&](const CanonicalActorView &actor,
          llvm::ArrayRef<unsigned> ordinals) -> llvm::Expected<bool> {
    for (unsigned ordinal : ordinals) {
      if (ordinal >= actor.op->getNumOperands() ||
          isMemoryCapability(actor.op->getOperand(ordinal).getType()))
        return false;
      auto producer = graphProducer(ActorTokenOperandRef{actor.ref, ordinal});
      if (!producer)
        return producer.takeError();
      auto fact = activityDefinedness(*producer);
      if (!fact)
        return fact.takeError();
      if (*fact != ActivityDefinedness::AlwaysDefined)
        return false;
    }
    return true;
  };

  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t actorSlot = 0; actorSlot < actors_.size(); ++actorSlot) {
      const CanonicalActorView &actor = actors_[actorSlot];
      const ActivityDefinednessTransferKind transfer =
          activityDefinednessTransfer(requireOperationSchema(actor.op));
      for (unsigned result = 0; result < actor.op->getNumResults(); ++result) {
        if (activityDefinednessByActorSlot_[actorSlot][result] ==
            ActivityDefinedness::AlwaysDefined)
          continue;

        llvm::SmallVector<unsigned, 4> dependencies;
        bool unconditional = false;
        switch (transfer) {
        case ActivityDefinednessTransferKind::Missing:
          continue;
        case ActivityDefinednessTransferKind::AlwaysDefined:
          unconditional = true;
          break;
        case ActivityDefinednessTransferKind::AllOperands:
          dependencies.resize_for_overwrite(actor.op->getNumOperands());
          std::iota(dependencies.begin(), dependencies.end(), 0U);
          break;
        case ActivityDefinednessTransferKind::FloatCompare: {
          auto comparison = llvm::cast<mlir::arith::CmpFOp>(actor.op);
          constexpr auto poisonFlags = mlir::arith::FastMathFlags::nnan |
                                       mlir::arith::FastMathFlags::ninf;
          if (mlir::arith::bitEnumContainsAny(comparison.getFastmath(),
                                              poisonFlags))
            continue;
          dependencies.resize_for_overwrite(actor.op->getNumOperands());
          std::iota(dependencies.begin(), dependencies.end(), 0U);
          break;
        }
        case ActivityDefinednessTransferKind::SameOrdinalOperand:
          dependencies.push_back(result);
          break;
        case ActivityDefinednessTransferKind::Parallelize:
          if (result == 0)
            dependencies = {0, 1};
          else
            dependencies = {1};
          break;
        case ActivityDefinednessTransferKind::Serialize:
          if (result == 0)
            dependencies = {0, 1, 2};
          else
            dependencies = {1, 2};
          break;
        }
        auto proved = unconditional ? llvm::Expected<bool>(true)
                                    : operandsAreDefined(actor, dependencies);
        if (!proved)
          return proved.takeError();
        if (!*proved)
          continue;
        activityDefinednessByActorSlot_[actorSlot][result] =
            ActivityDefinedness::AlwaysDefined;
        changed = true;
      }
    }
  }
  return llvm::Error::success();
}

} // namespace dataflow
