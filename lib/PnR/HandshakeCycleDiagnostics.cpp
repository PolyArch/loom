#include "HandshakeCycleDiagnostics.h"

#include "Common/MappingDebugLog.h"
#include "Fabric/Identity/FabricRefText.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <type_traits>

namespace loom::pnr::detail {
namespace {

llvm::StringRef signalKind(::loom::fabric::HandshakeSignalKind kind) {
  switch (kind) {
  case ::loom::fabric::HandshakeSignalKind::Valid:
    return "valid";
  case ::loom::fabric::HandshakeSignalKind::Ready:
    return "ready";
  }
  llvm_unreachable("unknown handshake signal kind");
}

llvm::StringRef ownerKind(::loom::fabric::FabricHandshakeOwnerKind kind) {
  using Kind = ::loom::fabric::FabricHandshakeOwnerKind;
  switch (kind) {
  case Kind::PointConnection:
    return "point_connection";
  case Kind::PeOccurrence:
    return "pe_occurrence";
  case Kind::FuOccurrence:
    return "fu_occurrence";
  case Kind::MemoryOccurrence:
    return "memory_occurrence";
  case Kind::SwitchOccurrence:
    return "switch_occurrence";
  case Kind::FifoOccurrence:
    return "fifo_occurrence";
  case Kind::BoundaryOccurrence:
    return "boundary_occurrence";
  case Kind::TransferPattern:
    return "transfer_pattern";
  }
  llvm_unreachable("unknown handshake owner kind");
}

void appendOwnerFields(llvm::json::Object &fields,
                       const ::loom::fabric::FabricHandshakeOwner &owner) {
  fields["owner_kind"] = ownerKind(owner.kind());
  std::visit(
      [&](const auto &payload) {
        using Payload = std::decay_t<decltype(payload)>;
        if constexpr (std::is_same_v<
                          Payload,
                          ::loom::fabric::FabricPointConnectionPayload>) {
          fields["source_endpoint_ref"] =
              ::loom::fabric::printFabricRef(payload.source);
          fields["destination_endpoint_ref"] =
              ::loom::fabric::printFabricRef(payload.destination);
        } else {
          fields["owner_ref"] = ::loom::fabric::printFabricRef(payload);
        }
      },
      owner.payload());
}

} // namespace

void emitHandshakeCycleDiagnostic(
    const FrozenSpatialHandshakeIndex &index, HandshakeCycleOrigin origin,
    llvm::ArrayRef<PnrIndex> frozenWitness,
    llvm::ArrayRef<PnrIndex> activeFragments,
    llvm::ArrayRef<PnrIndex> fragmentRefcounts) {
  using namespace ::loom::mapping_debug;
  const Level level = origin == HandshakeCycleOrigin::Candidate
                          ? Level::Decision
                          : Level::Detail;
  if (frozenWitness.empty() || !enabled(level))
    return;
  emit(level, Stage::SpatialPnr, Event::MappingFailure,
       [&](llvm::json::Object &fields) {
         fields["operation"] =
             origin == HandshakeCycleOrigin::Candidate
                 ? "selected_handshake_cycle"
                 : "projected_handshake_cycle";
         fields["arc_numbering"] = "frozen_projection";
         fields["witness_arc_count"] = frozenWitness.size();
         const std::size_t sampleCount =
             enabled(Level::Detail)
                 ? frozenWitness.size()
                 : std::min<std::size_t>(frozenWitness.size(), 8);
         fields["witness_arc_sample_count"] = sampleCount;
         fields["witness_arc_omitted_count"] = frozenWitness.size() - sampleCount;
         llvm::json::Array arcs;
         const auto projectedArcs = index.projectionArcs();
         const auto signals = index.projectionNodeSignals();
         const auto fragmentOffsets = index.projectionFragmentArcOffsets();
         const auto fragmentArcs = index.projectionFragmentArcs();
         const auto fragments = index.fragments();
         const auto models = index.ownerModels();
         for (PnrIndex arc : frozenWitness.take_front(sampleCount)) {
           const FrozenSpatialHandshakeArc &record = projectedArcs[arc];
           llvm::json::Object entry;
           entry["arc_ref"] = arc;
           entry["source_node"] = record.source;
           entry["destination_node"] = record.destination;
           if (signals[record.source]) {
             entry["source_endpoint_ref"] = ::loom::fabric::printFabricRef(
                 signals[record.source]->endpoint);
             entry["source_signal"] = signalKind(signals[record.source]->signal);
           }
           if (signals[record.destination]) {
             entry["destination_endpoint_ref"] = ::loom::fabric::printFabricRef(
                 signals[record.destination]->endpoint);
             entry["destination_signal"] =
                 signalKind(signals[record.destination]->signal);
           }
           llvm::json::Array contributors;
           std::size_t contributorCount = 0;
           for (PnrIndex fragment : activeFragments) {
             if (!llvm::is_contained(
                     fragmentArcs.slice(fragmentOffsets[fragment],
                                        fragmentOffsets[fragment + 1] -
                                            fragmentOffsets[fragment]),
                     arc))
               continue;
             ++contributorCount;
             if (!enabled(Level::Detail) && contributors.size() == 4)
               continue;
             llvm::json::Object contribution;
             contribution["fragment_ref"] = fragment;
             contribution["fragment_refcount"] = fragmentRefcounts[fragment];
             const PnrIndex owner = fragments[fragment].owner;
             contribution["owner_ordinal"] = owner;
             appendOwnerFields(contribution, models[owner].owner());
             contributors.push_back(std::move(contribution));
           }
           entry["active_contribution_count"] = contributorCount;
           entry["active_contribution_omitted_count"] =
               contributorCount - contributors.size();
           entry["active_contributions"] = std::move(contributors);
           arcs.push_back(std::move(entry));
         }
         fields["witness_arcs"] = std::move(arcs);
       });
}

} // namespace loom::pnr::detail
