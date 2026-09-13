#include "PnR/SpatialHandshakeSupplyDeficit.h"

#include "Common/MappingDebugLog.h"
#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialPnrProblem.h"

#include "Fabric/Identity/FabricHandshake.h"
#include "Fabric/Identity/FabricRefs.h"
#include "Fabric/IR/FabricEnums.h"

#include "SpatialProgressIndex.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <system_error>
#include <utility>
#include <variant>

using namespace loom::pnr;

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_handshake_supply_deficit_invalid: " + message);
}

std::vector<PnrIndex> canonical(llvm::ArrayRef<PnrIndex> values) {
  std::vector<PnrIndex> sorted(values.begin(), values.end());
  llvm::sort(sorted);
  sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
  return sorted;
}

/// Whether no Mapping selection on the current Fabric can isolate ready from
/// valid anywhere on the core. A buffered FIFO occurrence contributes no
/// cross-FIFO arc and a Temporal PE ingress contributes neither a
/// forward-valid nor a backward-ready arc, so neither complete isolation point
/// can lie on a cycle at all. A FIFO occurrence reaches a cycle only through a
/// bypass traversal, whose buffered alternative cuts both directions. A core
/// with no FIFO-occurrence contribution therefore has every remaining
/// ready-to-valid crossing inside one FU operation case or one switch input
/// row set, which no placement or route can open.
llvm::Expected<bool> coreLacksIsolation(
    const FrozenSpatialHandshakeIndex &index,
    llvm::ArrayRef<PnrIndex> coreArcs) {
  if (coreArcs.empty())
    return false;
  const auto arcs = index.projectionArcs();
  std::vector<std::uint64_t> coreMarks((arcs.size() + 63) / 64, 0);
  for (PnrIndex arc : coreArcs) {
    if (arc >= arcs.size())
      return invalid("cycle core names a foreign projection arc");
    coreMarks[arc / 64] |= std::uint64_t{1} << (arc % 64);
  }
  const auto onCore = [&](PnrIndex arc) {
    return (coreMarks[arc / 64] & (std::uint64_t{1} << (arc % 64))) != 0;
  };
  const auto fragments = index.fragments();
  const auto models = index.ownerModels();
  const auto fragmentOffsets = index.projectionFragmentArcOffsets();
  const auto fragmentArcs = index.projectionFragmentArcs();
  if (fragmentOffsets.size() != fragments.size() + 1)
    return invalid("fragment arc offsets disagree with the fragment domain");
  for (PnrIndex fragment = 0; fragment != fragments.size(); ++fragment) {
    const PnrIndex owner = fragments[fragment].owner;
    if (owner >= models.size())
      return invalid("handshake fragment names a foreign owner");
    if (models[owner].owner().kind() !=
        ::loom::fabric::FabricHandshakeOwnerKind::FifoOccurrence)
      continue;
    for (PnrIndex arc :
         fragmentArcs.slice(fragmentOffsets[fragment],
                            fragmentOffsets[fragment + 1] -
                                fragmentOffsets[fragment]))
      if (arc < arcs.size() && onCore(arc))
        return false;
  }
  return true;
}

} // namespace

void SpatialHandshakeCycleCore::reset() {
  witnesses_.clear();
  coreArcs_.clear();
  coreLogicalNets_.clear();
  classifiedArcs_.clear();
  lastArcs_.clear();
  metCycles_ = 0;
  coreRecurrences_ = 0;
  established_ = false;
}

llvm::Error SpatialHandshakeCycleCore::meet(
    const FrozenSpatialHandshakeIndex &index,
    llvm::ArrayRef<PnrIndex> frozenCycleArcs,
    llvm::ArrayRef<PnrIndex> contributingLogicalNets) {
  if (frozenCycleArcs.empty())
    return llvm::Error::success();
  ++metCycles_;
  std::vector<PnrIndex> arcs = canonical(frozenCycleArcs);
  const bool returned = arcs != lastArcs_;
  lastArcs_ = arcs;
  const auto insertion = llvm::lower_bound(
      witnesses_, arcs, [](const MetWitness &entry,
                           const std::vector<PnrIndex> &probe) {
        return entry.arcs < probe;
      });
  if (insertion == witnesses_.end() || insertion->arcs != arcs) {
    const auto inserted = witnesses_.insert(
        insertion, MetWitness{std::move(arcs),
                              canonical(contributingLogicalNets), 0});
    inserted->witnessCountAtLastMeeting = witnesses_.size();
    return llvm::Error::success();
  }
  const bool discovered = insertion->witnessCountAtLastMeeting !=
                          static_cast<std::uint64_t>(witnesses_.size());
  insertion->witnessCountAtLastMeeting = witnesses_.size();
  if (!returned || discovered)
    return llvm::Error::success();
  // The closure left this cycle, orbited among cycles it already knew, and
  // arrived back at it having found nothing new. That witness is the core the
  // search cannot leave behind.
  ++coreRecurrences_;
  coreArcs_ = arcs;
  coreLogicalNets_ = insertion->logicalNets;
  if (established_ || classifiedArcs_ == coreArcs_)
    return llvm::Error::success();
  classifiedArcs_ = coreArcs_;
  auto lacksIsolation = coreLacksIsolation(index, coreArcs_);
  if (!lacksIsolation)
    return lacksIsolation.takeError();
  established_ = *lacksIsolation;
  ::loom::mapping_debug::emit(
      ::loom::mapping_debug::Level::Summary,
      ::loom::mapping_debug::Stage::SpatialPnr,
      ::loom::mapping_debug::Event::MappingFailure,
      [&](llvm::json::Object &fields) {
        fields["operation"] = "recurring_handshake_cycle_core";
        fields["arc_numbering"] = "frozen_projection";
        fields["lacks_isolation"] = established_;
        fields["met_cycles"] = metCycles_;
        fields["distinct_cycle_witnesses"] =
            static_cast<std::uint64_t>(witnesses_.size());
        fields["core_recurrences"] = coreRecurrences_;
        fields["core_arc_count"] = static_cast<std::uint64_t>(coreArcs_.size());
        llvm::json::Array coreArcRefs;
        for (PnrIndex arc : coreArcs_)
          coreArcRefs.push_back(static_cast<std::int64_t>(arc));
        fields["core_arc_refs"] = std::move(coreArcRefs);
        llvm::json::Array coreNets;
        for (PnrIndex logicalNet : coreLogicalNets_)
          coreNets.push_back(static_cast<std::int64_t>(logicalNet));
        fields["core_logical_nets"] = std::move(coreNets);
      });
  return llvm::Error::success();
}

std::size_t SpatialHandshakeCycleCore::retainedStorageBytes() const {
  std::size_t bytes = witnesses_.capacity() * sizeof(MetWitness);
  for (const MetWitness &witness : witnesses_)
    bytes += (witness.arcs.capacity() + witness.logicalNets.capacity()) *
             sizeof(PnrIndex);
  return bytes + (coreArcs_.capacity() + coreLogicalNets_.capacity() +
                  classifiedArcs_.capacity() + lastArcs_.capacity()) *
                     sizeof(PnrIndex);
}

bool loom::pnr::spatialHandshakeCoreCoPlacementEscapes(
    const SpatialHandshakeCoreCoPlacement &coPlacement,
    const FrozenSpatialComputePlacement &placement) {
  return !llvm::is_contained(coPlacement.neighbourhood, placement.parentPe);
}

llvm::Expected<SpatialHandshakeCoreCoPlacement>
loom::pnr::projectSpatialHandshakeCoreCoPlacement(
    const SpatialCandidateState &candidate, llvm::ArrayRef<PnrIndex> coreArcs) {
  SpatialHandshakeCoreCoPlacement coPlacement;
  if (coreArcs.empty())
    return coPlacement;
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const FrozenSpatialHandshakeIndex &index = problem.handshake();
  const auto arcs = index.projectionArcs();
  std::vector<std::uint64_t> coreMarks((arcs.size() + 63) / 64, 0);
  for (PnrIndex arc : coreArcs) {
    if (arc >= arcs.size())
      return invalid("cycle core names a foreign projection arc");
    coreMarks[arc / 64] |= std::uint64_t{1} << (arc % 64);
  }
  const auto onCore = [&](PnrIndex arc) {
    return arc < arcs.size() &&
           (coreMarks[arc / 64] & (std::uint64_t{1} << (arc % 64))) != 0;
  };
  // The FU occurrences whose fragments contribute an arc of the core are the
  // Fabric side of the class; the placement domain supplies its PnR side.
  const auto fragments = index.fragments();
  const auto models = index.ownerModels();
  const auto fragmentOffsets = index.projectionFragmentArcOffsets();
  const auto fragmentArcs = index.projectionFragmentArcs();
  if (fragmentOffsets.size() != fragments.size() + 1)
    return invalid("fragment arc offsets disagree with the fragment domain");
  std::vector<::loom::fabric::FabricFuOccurrenceRef> coreFus;
  for (PnrIndex fragment = 0; fragment != fragments.size(); ++fragment) {
    const PnrIndex owner = fragments[fragment].owner;
    if (owner >= models.size())
      return invalid("handshake fragment names a foreign owner");
    if (models[owner].owner().kind() !=
        ::loom::fabric::FabricHandshakeOwnerKind::FuOccurrence)
      continue;
    const auto fu =
        std::get<::loom::fabric::FabricFuOccurrenceRef>(models[owner].owner().payload());
    if (llvm::is_contained(coreFus, fu))
      continue;
    for (PnrIndex arc :
         fragmentArcs.slice(fragmentOffsets[fragment],
                            fragmentOffsets[fragment + 1] -
                                fragmentOffsets[fragment]))
      if (onCore(arc)) {
        coreFus.push_back(fu);
        break;
      }
  }
  coPlacement.coreFuOccurrenceCount =
      static_cast<std::uint64_t>(coreFus.size());
  const auto placements = problem.realizations().computePlacements();
  for (PnrIndex placement = 0; placement != placements.size(); ++placement) {
    const FrozenSpatialComputePlacement &record = placements[placement];
    if (!llvm::is_contained(coreFus, record.fu))
      continue;
    if (candidate.computeBinding(record.realization).placement != placement)
      continue;
    if (!llvm::is_contained(coPlacement.computeDecisions, record.realization))
      coPlacement.computeDecisions.push_back(record.realization);
    if (!llvm::is_contained(coPlacement.neighbourhood, record.parentPe))
      coPlacement.neighbourhood.push_back(record.parentPe);
  }
  llvm::sort(coPlacement.computeDecisions);
  const auto realizations = problem.realizations().computeRealizations();
  for (PnrIndex decision : coPlacement.computeDecisions) {
    if (decision >= realizations.size())
      return invalid("cycle core names a foreign compute realization");
    const FrozenSpatialComputeRealization &realization = realizations[decision];
    for (PnrIndex option = realization.placementOffset;
         option != realization.placementOffset + realization.placementCount;
         ++option) {
      if (option >= placements.size())
        return invalid("compute realization placement is out of range");
      if (spatialHandshakeCoreCoPlacementEscapes(coPlacement,
                                                 placements[option])) {
        coPlacement.escapable = true;
        break;
      }
    }
    if (coPlacement.escapable)
      break;
  }
  return coPlacement;
}

SpatialHandshakeCycleCoreSummary loom::pnr::summarizeSpatialHandshakeCycleCores(
    llvm::ArrayRef<const SpatialHandshakeCycleCore *> cores) {
  SpatialHandshakeCycleCoreSummary summary;
  for (const SpatialHandshakeCycleCore *core : cores) {
    summary.metCycles += core->metCycles();
    summary.distinctWitnesses += core->distinctWitnesses();
    summary.coreRecurrences += core->coreRecurrences();
    if (!summary.established && core->established())
      summary.established = core;
  }
  return summary;
}

llvm::Expected<std::optional<SpatialFifoCapacitySuggestion>>
loom::pnr::projectSpatialHandshakeSupplyDeficit(
    const SpatialCandidateState &candidate,
    const SpatialHandshakeCycleCore &core) {
  if (!core.established())
    return std::optional<SpatialFifoCapacitySuggestion>();
  // A core whose co-placement class the exact repair can still state and
  // break is a search fact, not a supply fact: the Fabric already offers the
  // Temporal ingress or the second neighbourhood that opens it, and growing
  // reserved channels would not be the isolation it lacks.
  auto coPlacement =
      projectSpatialHandshakeCoreCoPlacement(candidate, core.arcs());
  if (!coPlacement)
    return coPlacement.takeError();
  ::loom::mapping_debug::emit(
      ::loom::mapping_debug::Level::Summary,
      ::loom::mapping_debug::Stage::SpatialPnr,
      ::loom::mapping_debug::Event::MappingFailure,
      [&](llvm::json::Object &fields) {
        fields["operation"] = "handshake_cycle_core_isolation";
        fields["class_decision_count"] =
            static_cast<std::uint64_t>(coPlacement->computeDecisions.size());
        fields["neighbourhood_pe_count"] =
            static_cast<std::uint64_t>(coPlacement->neighbourhood.size());
        fields["core_fu_occurrence_count"] =
            coPlacement->coreFuOccurrenceCount;
        fields["class_escapable"] = coPlacement->escapable;
        fields["lacking_isolation"] =
            !coPlacement->computeDecisions.empty() && !coPlacement->escapable
                ? "temporal_pe_ingress_or_second_neighbourhood"
                : coPlacement->computeDecisions.empty()
                      ? "switch_row_set_has_no_compute_class"
                      : "none";
      });
  if (!coPlacement->computeDecisions.empty() && coPlacement->escapable)
    return std::optional<SpatialFifoCapacitySuggestion>();

  // The recipe owns one reserved-channel guarantee for the interconnect, so
  // the binding guarantee is the largest a tag-selective owner declares. A
  // strict queue grants exclusive use regardless of any reservation count and
  // has no channel proposal, exactly as the capacity-shortfall owner requires.
  const detail::FrozenSpatialProgressIndex &progress =
      candidate.problem().progressIndex();
  const auto owners = progress.finiteBufferOwners();
  const auto capacities = progress.ownerGuaranteedNetCapacities();
  const auto disciplines = progress.ownerQueueDisciplines();
  if (capacities.size() != owners.size() || disciplines.size() != owners.size())
    return invalid("finite-buffer owner projections disagree in width");
  std::optional<PnrIndex> binding;
  for (PnrIndex owner = 0; owner != owners.size(); ++owner) {
    if (disciplines[owner] != ::fabric::FifoQueueDiscipline::PerTagVirtualChannel)
      continue;
    if (capacities[owner] == 0)
      continue;
    if (!binding || capacities[owner] > capacities[*binding])
      binding = owner;
  }
  if (!binding)
    return std::optional<SpatialFifoCapacitySuggestion>();
  const std::uint64_t selected = capacities[*binding];
  if (selected == std::numeric_limits<std::uint64_t>::max())
    return invalid("reserved-channel guarantee cannot be raised");

  SpatialFifoCapacitySuggestion deficit;
  deficit.owner = owners[*binding];
  deficit.selectedCapacity = selected;
  deficit.sufficientCapacity = selected + 1;
  const auto logicalNets = candidate.problem().transfers().logicalNets();
  deficit.logicalNets.reserve(core.logicalNets().size());
  for (PnrIndex logicalNet : core.logicalNets()) {
    if (logicalNet >= logicalNets.size())
      return invalid("cycle core names a foreign logical net");
    deficit.logicalNets.push_back(logicalNets[logicalNet].producer);
  }
  const auto traversals = candidate.problem().routing().traversals();
  for (PnrIndex traversal : progress.traversalsForOwner(*binding)) {
    if (traversal >= traversals.size())
      return invalid("finite-buffer owner names a foreign traversal");
    deficit.routeAnchors.push_back(traversals[traversal].reference);
  }
  return std::optional<SpatialFifoCapacitySuggestion>(std::move(deficit));
}
