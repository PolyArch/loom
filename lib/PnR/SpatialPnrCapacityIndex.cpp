#include "SpatialPnrCapacityIndex.h"

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/TemporalPeResourceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefText.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::fabric;
using namespace loom::mapping;
using namespace loom::pnr;

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::make_error<SpatialPnrFreezeFailure>(
      SpatialPnrFreezeFailureKind::Invalid, message.str());
}

std::string refKey(const FabricUsePatternRef &reference) {
  const std::vector<std::uint8_t> bytes = canonicalFabricBytes(reference);
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

llvm::Expected<PnrIndex>
patternOrdinal(const llvm::StringMap<PnrIndex> &patternByRef,
               const FabricUsePatternRef &reference) {
  const auto found = patternByRef.find(refKey(reference));
  if (found == patternByRef.end())
    return invalid("selected ResourceUse names absent Fabric UsePattern " +
                   printFabricRef(reference));
  return found->second;
}

llvm::Expected<std::uint64_t>
atomicEnvelopeOveruse(const FrozenSpatialResourceIndex &resources,
                      llvm::ArrayRef<PnrIndex> patterns) {
  llvm::SmallDenseMap<PnrIndex, std::uint64_t, 8> demand;
  for (PnrIndex patternOrdinal : patterns) {
    if (patternOrdinal >= resources.usePatterns().size())
      return invalid("atomic capacity envelope contains an invalid pattern");
    const FrozenSpatialUsePattern &pattern =
        resources.usePatterns()[patternOrdinal];
    for (const FrozenSpatialResourceClaim &claim :
         resources.claims().slice(pattern.claimOffset, pattern.claimCount)) {
      if (claim.state >= resources.resourceStates().size())
        return invalid("atomic capacity envelope contains an invalid state");
      const FrozenSpatialResourceState &state =
          resources.resourceStates()[claim.state];
      if (claim.dimension >= state.capacityCount)
        return invalid(
            "atomic capacity envelope contains an invalid dimension");
      const PnrIndex dimension = state.capacityOffset + claim.dimension;
      std::uint64_t &amount = demand[dimension];
      if (claim.amount > std::numeric_limits<std::uint64_t>::max() - amount)
        return invalid("atomic capacity demand overflows u64");
      amount += claim.amount;
    }
  }

  std::uint64_t total = 0;
  for (const auto &entry : demand) {
    if (entry.first >= resources.capacityDimensions().size())
      return invalid("atomic capacity envelope resolved a foreign dimension");
    const FrozenSpatialCapacityDimension &dimension =
        resources.capacityDimensions()[entry.first];
    if (entry.second >
        std::numeric_limits<std::uint64_t>::max() - dimension.initialOccupancy)
      return invalid("atomic capacity usage overflows u64");
    const std::uint64_t usage = dimension.initialOccupancy + entry.second;
    const std::uint64_t overuse =
        usage > dimension.capacity ? usage - dimension.capacity : 0;
    if (overuse > std::numeric_limits<std::uint64_t>::max() - total)
      return invalid("atomic capacity overuse overflows u64");
    total += overuse;
  }
  return total;
}

llvm::Expected<std::optional<FabricFuTemplatePortRef>>
consumedBoundary(const TechComputeRealizationView &realization,
                 dataflow::ActorRef actor, std::uint32_t operand) {
  std::optional<FabricFuTemplatePortRef> result;
  for (const TechComputeBoundaryView &boundary : realization.boundaries) {
    if (boundary.actor != actor ||
        boundary.direction != FabricPortDirection::Input ||
        boundary.portOrdinal != operand)
      continue;
    if (result)
      return invalid("one actor input has duplicate FU boundary witnesses");
    result = boundary.fabricPort;
  }
  return result;
}

llvm::Error checkedAccumulate(std::uint64_t value, std::uint64_t &total,
                              llvm::StringRef subject) {
  if (value > std::numeric_limits<std::uint64_t>::max() - total)
    return invalid(subject + " overflows u64");
  total += value;
  return llvm::Error::success();
}

} // namespace

class loom::pnr::FrozenSpatialCapacityIndexBuilder final {
public:
  static llvm::Expected<FrozenSpatialCapacityIndex>
  build(const dataflow::CanonicalDataflowProgramView &dataflow,
        const TechMappingView &techMapping, const FabricArtifactView &fabric,
        const FrozenSpatialRealizationIndex &realizations,
        const FrozenSpatialResourceIndex &resources,
        const FrozenSpatialRoutingGraph &routing,
        const FrozenSpatialHandshakeIndex &handshake) {
    llvm::StringMap<PnrIndex> patternByRef;
    for (auto [ordinal, pattern] : llvm::enumerate(resources.usePatterns())) {
      const auto inserted = patternByRef.try_emplace(
          refKey(pattern.reference), static_cast<PnrIndex>(ordinal));
      if (!inserted.second)
        return invalid("frozen Fabric UsePattern inventory is not unique");
    }

    FrozenSpatialCapacityIndex result;
    result.computeInstructionContextOveruse_.assign(
        realizations.computeInstructionContexts().size(), 0);
    for (auto [realizationOrdinal, frozenRealization] :
         llvm::enumerate(realizations.computeRealizations())) {
      if (realizationOrdinal >= techMapping.computeRealizations().size())
        return invalid("compute capacity owner is absent from TechMapping");
      const TechComputeRealizationView &realization =
          techMapping.computeRealizations()[realizationOrdinal];

      for (const FrozenSpatialComputePlacement &placement :
           realizations.computePlacements().slice(
               frozenRealization.placementOffset,
               frozenRealization.placementCount)) {
        for (PnrIndex contextOrdinal = placement.contextOffset;
             contextOrdinal != placement.contextOffset + placement.contextCount;
             ++contextOrdinal) {
          const InstructionContextRef context =
              realizations.computeInstructionContexts()[contextOrdinal];
          std::uint64_t placementOveruse = 0;

          for (const TechComputeActorView &binding : realization.actors) {
            auto actor = dataflow.resolve(binding.actor);
            if (!actor)
              return actor.takeError();
            auto projection =
                dataflow::projectRegisteredActorSchemaProjection(actor->op);
            if (!projection)
              return projection.takeError();
            if (binding.operandPorts.size() >
                    std::numeric_limits<std::uint32_t>::max() ||
                binding.resultPorts.size() >
                    std::numeric_limits<std::uint32_t>::max())
              return invalid("compute actor port domain exceeds u32");
            auto cases = dataflow::semantics::projectActorHandshakeCases(
                projection->schema,
                static_cast<std::uint32_t>(binding.operandPorts.size()),
                static_cast<std::uint32_t>(binding.resultPorts.size()));
            if (!cases)
              return cases.takeError();

            const ResolvedFabricOpCapabilityView *capability =
                fabric.resolvedFabricOpCapability(binding.fabricOperation);
            if (!capability)
              return invalid("compute actor has no resolved fabric.op");
            auto occurrenceOperation = deriveFabricFuOccurrenceNode(
                fabric, binding.fabricOperation, placement.fu);
            if (!occurrenceOperation)
              return occurrenceOperation.takeError();
            const FabricInventoryOwnerRef operationOwner =
                FabricInventoryOwnerRef::of(*occurrenceOperation);

            for (const dataflow::semantics::ActorHandshakeCase &transition :
                 *cases) {
              auto operationPattern = ::fabric::resolveOperationUsePattern(
                  capability->resourceStateAndTimingContract,
                  transition.ordinal);
              if (!operationPattern)
                return operationPattern.takeError();
              const FabricUsePatternRef operationPatternRef{
                  FabricUsePatternOwnerRef(operationOwner),
                  operationPattern->ordinal()};
              auto denseOperationPattern =
                  patternOrdinal(patternByRef, operationPatternRef);
              if (!denseOperationPattern)
                return denseOperationPattern.takeError();
              llvm::SmallVector<PnrIndex, 8> envelope{*denseOperationPattern};

              if (fabric.peSchedule(placement.parentPe) ==
                  ::fabric::Schedule::Temporal) {
                for (std::uint32_t operand : transition.consumedInputs) {
                  auto boundary =
                      consumedBoundary(realization, binding.actor, operand);
                  if (!boundary)
                    return boundary.takeError();
                  if (!*boundary)
                    continue;
                  auto queuePattern =
                      ::fabric::resolveTemporalPeOperandQueuePattern(
                          fabric, context, placement.fu, (**boundary).ordinal,
                          ::fabric::TemporalOperandQueueUse::Dequeue);
                  if (!queuePattern)
                    return queuePattern.takeError();
                  auto denseQueuePattern =
                      patternOrdinal(patternByRef, *queuePattern);
                  if (!denseQueuePattern)
                    return denseQueuePattern.takeError();
                  envelope.push_back(*denseQueuePattern);
                }
              }

              auto overuse = atomicEnvelopeOveruse(resources, envelope);
              if (!overuse)
                return overuse.takeError();
              if (llvm::Error error =
                      checkedAccumulate(*overuse, placementOveruse,
                                        "compute atomic capacity overuse"))
                return std::move(error);
            }
          }
          result.computeInstructionContextOveruse_[contextOrdinal] =
              placementOveruse;
        }
      }
    }

    result.memoryOperationPlanOveruse_.reserve(
        handshake.memoryOperationPlans().size());
    for (const FrozenSpatialMemoryOperationHandshakePlan &plan :
         handshake.memoryOperationPlans()) {
      const PnrIndex selected[] = {plan.usePattern};
      auto overuse = atomicEnvelopeOveruse(resources, selected);
      if (!overuse)
        return overuse.takeError();
      result.memoryOperationPlanOveruse_.push_back(*overuse);
    }

    // A traversal activation group is one owner-normalized physical use. It
    // must be individually feasible, while contention between independent
    // message events remains negotiated routing or ResourceTimeOverbooking.
    for (const FrozenSpatialRouteClaim &claim : routing.routeClaims()) {
      if (claim.capacityDimension >= resources.capacityDimensions().size())
        return invalid("route claim names an invalid capacity dimension");
      const FrozenSpatialCapacityDimension &dimension =
          resources.capacityDimensions()[claim.capacityDimension];
      const std::uint64_t usage =
          static_cast<std::uint64_t>(dimension.initialOccupancy) + claim.amount;
      if (usage > dimension.capacity)
        return invalid(
            "one traversal activation group exceeds Fabric capacity");
    }
    return result;
  }
};

llvm::Expected<FrozenSpatialCapacityIndex>
loom::pnr::detail::buildFrozenSpatialCapacityIndex(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping, const FabricArtifactView &fabric,
    const FrozenSpatialRealizationIndex &realizations,
    const FrozenSpatialResourceIndex &resources,
    const FrozenSpatialRoutingGraph &routing,
    const FrozenSpatialHandshakeIndex &handshake) {
  return FrozenSpatialCapacityIndexBuilder::build(dataflow, techMapping, fabric,
                                                  realizations, resources,
                                                  routing, handshake);
}
