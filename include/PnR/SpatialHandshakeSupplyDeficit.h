#ifndef LOOM_PNR_SPATIALHANDSHAKESUPPLYDEFICIT_H
#define LOOM_PNR_SPATIALHANDSHAKESUPPLYDEFICIT_H

#include "Fabric/Identity/FabricRefs.h"
#include "PnR/PnrIndex.h"
#include "PnR/SpatialProgressState.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::pnr {

struct FrozenSpatialComputePlacement;
class FrozenSpatialHandshakeIndex;
class SpatialCandidateState;

/// The selected-handshake cycle core one Spatial routing closure keeps
/// meeting.
///
/// A closure *meets* a cycle whenever it keeps a selection whose provisional
/// projection is still cyclic: it retained a joint route trial that did not
/// open the cycle, or it gave up while the projection was cyclic. Either way
/// it names the frozen projection arcs of the witness it could not remove, and
/// the restart retains them.
///
/// A closure that is still learning meets a cycle it has not seen before. A
/// closure on a treadmill orbits among cycles it already knows: it leaves one
/// witness, meets others it has already met, discovers nothing new, and
/// arrives back at the first. That closed orbit is the recurrence this owner
/// recognises, and it is a property of the witness set alone: a return to a
/// witness with no witness discovered since that witness was last met. A first
/// meeting, a repeated meeting of the witness already in hand, and a return
/// that follows a discovery all fail it. No count, share, ratio, or elapsed
/// time takes part.
class SpatialHandshakeCycleCore final {
public:
  void reset();

  /// Records one met cycle. The arcs are ordinals of
  /// `FrozenSpatialHandshakeIndex::projectionArcs()`, and the logical nets are
  /// the transfer ordinals whose selected routes contributed them. A witness
  /// that closes an orbit becomes the retained core and is classified against
  /// the Fabric's isolation points once.
  llvm::Error meet(const FrozenSpatialHandshakeIndex &index,
                   llvm::ArrayRef<PnrIndex> frozenCycleArcs,
                   llvm::ArrayRef<PnrIndex> contributingLogicalNets);

  /// The recurring witness, empty until an orbit closes.
  llvm::ArrayRef<PnrIndex> arcs() const { return coreArcs_; }
  llvm::ArrayRef<PnrIndex> logicalNets() const { return coreLogicalNets_; }
  /// Whether a recurring core has no isolation left on this Fabric. This is
  /// the established Spatial supply deficit.
  bool established() const { return established_; }
  std::uint64_t metCycles() const { return metCycles_; }
  std::uint64_t distinctWitnesses() const { return witnesses_.size(); }
  std::uint64_t coreRecurrences() const { return coreRecurrences_; }
  std::size_t retainedStorageBytes() const;

private:
  /// One met witness: its sorted frozen arcs, the logical nets that
  /// contributed them, and how many distinct witnesses the restart had met
  /// when this one was last met.
  struct MetWitness final {
    std::vector<PnrIndex> arcs;
    std::vector<PnrIndex> logicalNets;
    std::uint64_t witnessCountAtLastMeeting = 0;
  };

  std::vector<MetWitness> witnesses_;
  std::vector<PnrIndex> coreArcs_;
  std::vector<PnrIndex> coreLogicalNets_;
  std::vector<PnrIndex> classifiedArcs_;
  std::vector<PnrIndex> lastArcs_;
  std::uint64_t metCycles_ = 0;
  std::uint64_t coreRecurrences_ = 0;
  bool established_ = false;
};

/// The co-placement class a recurring core names.
///
/// Every arc of the core is contributed by a fragment of an FU, PE, or switch
/// occurrence. The compute placements whose fragments contribute those arcs
/// are the core's actors, and the PE occurrences those placements sit on are
/// its *neighbourhood*: a region the interconnect leaves FIFO-free, so a route
/// between two of its members crosses no buffered mesh link. The crossings the
/// core closes are the multi-result gating of one FU operation case and the
/// multicast gating of one switch input row set, and both are properties of
/// that neighbourhood rather than of one occurrence in it.
///
/// A choice *escapes* the class when it binds the actor outside that
/// neighbourhood. Leaving is the only placement change that reaches an
/// isolation point: a Temporal PE ingress isolates a value leaving its own PE,
/// and a buffered mesh FIFO isolates a route leaving the tile, while every
/// placement inside the neighbourhood keeps both crossings combinational. A
/// class with no escaping choice is one the Fabric cannot open at all.
struct SpatialHandshakeCoreCoPlacement final {
  /// Compute decision ordinals, canonically ordered. Empty when no compute
  /// placement contributes an arc, which makes the class vacuous.
  std::vector<PnrIndex> computeDecisions;
  /// The Spatial PE occurrences the core's actors currently share.
  std::vector<::loom::fabric::FabricPeOccurrenceRef> neighbourhood;
  /// FU occurrences whose fragments contribute an arc of the core.
  std::uint64_t coreFuOccurrenceCount = 0;
  /// Whether some decision of the class still has an escaping choice.
  bool escapable = false;
};

llvm::Expected<SpatialHandshakeCoreCoPlacement>
projectSpatialHandshakeCoreCoPlacement(const SpatialCandidateState &candidate,
                                       llvm::ArrayRef<PnrIndex> coreArcs);

/// Whether one compute placement escapes the class.
bool spatialHandshakeCoreCoPlacementEscapes(
    const SpatialHandshakeCoreCoPlacement &coPlacement,
    const FrozenSpatialComputePlacement &placement);

/// What one restart's closure owners witnessed about selected handshake
/// cycles, and the first of them that established a supply deficit.
struct SpatialHandshakeCycleCoreSummary final {
  std::uint64_t metCycles = 0;
  std::uint64_t distinctWitnesses = 0;
  std::uint64_t coreRecurrences = 0;
  const SpatialHandshakeCycleCore *established = nullptr;
};

/// Reduces a restart's closure owners in their stage order.
SpatialHandshakeCycleCoreSummary summarizeSpatialHandshakeCycleCores(
    llvm::ArrayRef<const SpatialHandshakeCycleCore *> cores);

/// The typed Spatial supply deficit an established core publishes: one more
/// guaranteed channel than the binding reserved-channel guarantee of the
/// interconnect the core's routes negotiate, so the router gains the buffered
/// link it needs to cut the core. Absent when no core is established or when
/// the Fabric declares no tag-selective reserved-channel guarantee to raise.
llvm::Expected<std::optional<SpatialFifoCapacitySuggestion>>
projectSpatialHandshakeSupplyDeficit(const SpatialCandidateState &candidate,
                                     const SpatialHandshakeCycleCore &core);

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALHANDSHAKESUPPLYDEFICIT_H
