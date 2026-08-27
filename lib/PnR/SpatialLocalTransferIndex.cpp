#include "SpatialLocalTransferIndex.h"

#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"
#include "PnR/PnrIndex.h"
#include "PnR/SpatialCandidateState.h"
#include "SpatialRouteConstraintModel.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <system_error>
#include <vector>

using namespace loom;
using namespace loom::mapping;
using namespace loom::pnr;

namespace {

constexpr llvm::StringLiteral frozenArtifact = "FrozenSpatialPnrProblem";
constexpr PnrCapacityContext domainCountContext{
    frozenArtifact, "register_fifo_transfer_domains", "logical_nets",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext optionOffsetContext{
    frozenArtifact, "register_fifo_transfer_domains",
    "register_fifo_transfer_options", PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext optionCountContext{
    frozenArtifact, "register_fifo_transfer_options",
    "register_fifo_transfer_options", PnrCapacityMeasure::Count};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::make_error<SpatialPnrFreezeFailure>(
      SpatialPnrFreezeFailureKind::Invalid, message.str());
}

llvm::Expected<PnrIndex> checked(PnrCapacityContext context,
                                 std::size_t value) {
  return checkedPnrIndex(context, static_cast<std::uint64_t>(value));
}

const TechComputeRealizationView *
findRealization(const TechMappingView &techMapping,
                const ::dataflow::ActorRef &actor) {
  const TechComputeRealizationView *result = nullptr;
  for (const TechComputeRealizationView &candidate :
       techMapping.computeRealizations()) {
    if (!llvm::any_of(candidate.actors, [&](const auto &member) {
          return member.actor == actor;
        }))
      continue;
    if (result)
      return nullptr;
    result = &candidate;
  }
  return result;
}

std::optional<PnrIndex>
findFrozenRealization(const FrozenSpatialRealizationIndex &realizations,
                      std::uint64_t entity) {
  for (auto [ordinal, realization] :
       llvm::enumerate(realizations.computeRealizations()))
    if (realization.reference.entity == entity)
      return static_cast<PnrIndex>(ordinal);
  return std::nullopt;
}

} // namespace

class loom::pnr::FrozenSpatialLocalTransferIndexBuilder final {
public:
  static llvm::Expected<FrozenSpatialLocalTransferIndex>
  build(const ::dataflow::CanonicalDataflowProgramView &dataflow,
        const TechMappingView &techMapping,
        const ::loom::fabric::FabricArtifactView &fabric,
        const FrozenSpatialRealizationIndex &realizations,
        const FrozenSpatialTransferIndex &transfers,
        const FrozenSpatialRoutingGraph &routing) {
    FrozenSpatialLocalTransferIndex result;
    const auto logicalNets = transfers.logicalNets();
    if (llvm::Error error =
            preflightPnrIndexCapacity(domainCountContext, logicalNets.size()))
      return std::move(error);
    if (logicalNets.size() != techMapping.residualLogicalNets().size())
      return invalid("local-transfer net domain disagrees with TechMapping");

    result.domains_.reserve(logicalNets.size());
    for (auto [netOrdinalValue, logicalNet] : llvm::enumerate(logicalNets)) {
      const PnrIndex netOrdinal = static_cast<PnrIndex>(netOrdinalValue);
      auto offset = checked(optionOffsetContext, result.options_.size());
      if (!offset)
        return offset.takeError();
      const auto &sourceNet = techMapping.residualLogicalNets()[netOrdinal];
      if (sourceNet.producer != logicalNet.producer ||
          sourceNet.sinks.size() != logicalNet.sinkCount)
        return invalid("local-transfer net ordering is not canonical");

      const auto *producer =
          std::get_if<::dataflow::ActorTokenResultRef>(&sourceNet.producer);
      const auto *consumer =
          sourceNet.sinks.size() == 1
              ? std::get_if<::dataflow::ActorTokenOperandRef>(
                    &sourceNet.sinks.front())
              : nullptr;
      if (producer && consumer) {
        const TechComputeRealizationView *producerRealization =
            findRealization(techMapping, producer->actor);
        const TechComputeRealizationView *consumerRealization =
            findRealization(techMapping, consumer->actor);
        if (producerRealization && consumerRealization) {
          auto producerOrdinal = findFrozenRealization(
              realizations, producerRealization->entityId);
          auto consumerOrdinal = findFrozenRealization(
              realizations, consumerRealization->entityId);
          if (!producerOrdinal || !consumerOrdinal)
            return invalid("local-transfer realization was not frozen");
          if (llvm::Error error = appendPlacementPairs(
                  dataflow, *producerRealization, *consumerRealization, fabric,
                  realizations, routing, sourceNet,
                  netOrdinal, *producerOrdinal, *consumerOrdinal, result))
            return std::move(error);
        }
      }

      auto count =
          checked(optionCountContext, result.options_.size() - *offset);
      if (!count)
        return count.takeError();
      result.domains_.push_back({*offset, *count});
    }
    return result;
  }

private:
  static llvm::Error
  appendPlacementPairs(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                       const TechComputeRealizationView &producerTech,
                       const TechComputeRealizationView &consumerTech,
                       const ::loom::fabric::FabricArtifactView &fabric,
                       const FrozenSpatialRealizationIndex &realizations,
                       const FrozenSpatialRoutingGraph &routing,
                       const TechResidualLogicalNetView &logicalNet,
                       PnrIndex logicalNetOrdinal, PnrIndex producerRealization,
                       PnrIndex consumerRealization,
                       FrozenSpatialLocalTransferIndex &result) {
    const auto realizationRecords = realizations.computeRealizations();
    const auto placements = realizations.computePlacements();
    const auto contexts = realizations.computeInstructionContexts();
    const auto &producerRecord = realizationRecords[producerRealization];
    const auto &consumerRecord = realizationRecords[consumerRealization];

    for (PnrIndex producerPlacement = producerRecord.placementOffset;
         producerPlacement !=
         producerRecord.placementOffset + producerRecord.placementCount;
         ++producerPlacement) {
      const auto &producerChoice = placements[producerPlacement];
      const auto producerPe = fabric.parentPeOf(producerChoice.fu);
      if (!producerPe)
        return invalid("local-transfer producer FU has no parent PE");
      const auto producerSchedule = fabric.peSchedule(*producerPe);
      if (!producerSchedule)
        return invalid("local-transfer producer PE has no schedule");
      if (*producerSchedule != ::fabric::Schedule::Temporal)
        continue;
      for (PnrIndex consumerPlacement = consumerRecord.placementOffset;
           consumerPlacement !=
           consumerRecord.placementOffset + consumerRecord.placementCount;
           ++consumerPlacement) {
        if (producerRealization == consumerRealization &&
            producerPlacement != consumerPlacement)
          continue;
        const auto &consumerChoice = placements[consumerPlacement];
        const auto consumerPe = fabric.parentPeOf(consumerChoice.fu);
        if (!consumerPe)
          return invalid("local-transfer consumer FU has no parent PE");
        if (*consumerPe != *producerPe)
          continue;
        if (producerChoice.contextCount == 0 ||
            consumerChoice.contextCount == 0 ||
            producerChoice.contextOffset >= contexts.size() ||
            consumerChoice.contextOffset >= contexts.size())
          return invalid("local-transfer placement has no instruction context");

        std::vector<SpatialComputeBindingView> bindings;
        bindings.push_back({producerRecord.reference.entity,
                            producerChoice.fu,
                            contexts[producerChoice.contextOffset],
                            {}});
        if (consumerRealization != producerRealization)
          bindings.push_back({consumerRecord.reference.entity,
                              consumerChoice.fu,
                              contexts[consumerChoice.contextOffset],
                              {}});
        auto options = deriveSpatialPeLocalTransferOptionsForRealizations(
            dataflow, producerTech, consumerTech, fabric, bindings, logicalNet);
        if (!options)
          return options.takeError();
        for (const SpatialPeLocalTransferOptionView &option : *options) {
          const auto write =
              routing.topology().traversalOrdinal(option.writeTraversal);
          if (!write)
            return invalid("register-FIFO path is absent from the routing graph");
          const auto read =
              routing.topology().traversalOrdinal(option.readTraversal);
          if (!read)
            return invalid("register-FIFO path is absent from the routing graph");
          if (*write >= routing.traversals().size() ||
              *read >= routing.traversals().size())
            return invalid("local-transfer traversal ordinal is out of range");
          if (llvm::Error error = preflightPnrIndexCapacity(
                  optionCountContext, result.options_.size() + 1))
            return error;
          result.options_.push_back(FrozenSpatialRegisterFifoTransferOption{
              logicalNetOrdinal, producerRealization, consumerRealization,
              producerPlacement, consumerPlacement, option.pe,
              option.registerFifo, *write, *read, option.tag});
        }
      }
    }
    return llvm::Error::success();
  }
};

llvm::Expected<FrozenSpatialLocalTransferIndex>
loom::pnr::detail::buildFrozenSpatialLocalTransferIndex(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const FrozenSpatialRealizationIndex &realizations,
    const FrozenSpatialTransferIndex &transfers,
    const FrozenSpatialRoutingGraph &routing) {
  return FrozenSpatialLocalTransferIndexBuilder::build(
      dataflow, techMapping, fabric, realizations, transfers, routing);
}

namespace {

bool matchesBindings(
    const FrozenSpatialRegisterFifoTransferOption &option,
    llvm::ArrayRef<SpatialComputeBindingSelection> computeBindings) {
  return option.producerRealization < computeBindings.size() &&
         option.consumerRealization < computeBindings.size() &&
         computeBindings[option.producerRealization].placement ==
             option.producerPlacement &&
         computeBindings[option.consumerRealization].placement ==
             option.consumerPlacement;
}

using FifoKey =
    std::pair<::loom::fabric::FabricEntityId, ::loom::fabric::FabricOrdinal>;

} // namespace

llvm::Expected<std::optional<PnrIndex>>
loom::pnr::detail::findPreferredAvailableSpatialLocalTransfer(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<SpatialComputeBindingSelection> computeBindings,
    llvm::ArrayRef<PnrIndex> selections, PnrIndex logicalNet) {
  if (logicalNet >= problem.localTransfers().domains().size() ||
      computeBindings.size() !=
          problem.realizations().computeRealizations().size() ||
      selections.size() != problem.transfers().logicalNets().size())
    return invalid("local-transfer selection dimensions are inconsistent");
  if (problem.routeConstraints().netHasConstraints(logicalNet))
    return std::optional<PnrIndex>{};

  std::set<FifoKey> occupied;
  const auto options = problem.localTransfers().options();
  for (PnrIndex net = 0; net < selections.size(); ++net) {
    if (net == logicalNet || selections[net] == getInvalidPnrIndex())
      continue;
    if (selections[net] >= options.size())
      return invalid("selected local-transfer option is out of range");
    const auto &selected = options[selections[net]];
    occupied.emplace(selected.pe.id(), selected.registerFifo);
  }

  const auto &domain = problem.localTransfers().domains()[logicalNet];
  for (PnrIndex option = domain.optionOffset;
       option != domain.optionOffset + domain.optionCount; ++option) {
    const auto &candidate = options[option];
    if (matchesBindings(candidate, computeBindings) &&
        occupied.find({candidate.pe.id(), candidate.registerFifo}) ==
            occupied.end())
      return std::optional<PnrIndex>(option);
  }
  return std::optional<PnrIndex>{};
}

llvm::Expected<std::vector<PnrIndex>>
loom::pnr::detail::derivePreferredSpatialLocalTransferSelections(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<SpatialComputeBindingSelection> computeBindings) {
  std::vector<PnrIndex> selections(problem.transfers().logicalNets().size(),
                                   getInvalidPnrIndex());
  for (PnrIndex logicalNet = 0; logicalNet < selections.size(); ++logicalNet) {
    auto selected = findPreferredAvailableSpatialLocalTransfer(
        problem, computeBindings, selections, logicalNet);
    if (!selected)
      return selected.takeError();
    if (*selected)
      selections[logicalNet] = **selected;
  }
  return selections;
}
