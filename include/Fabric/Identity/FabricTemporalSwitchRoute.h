#ifndef LOOM_FABRIC_IDENTITY_FABRICTEMPORALSWITCHROUTE_H
#define LOOM_FABRIC_IDENTITY_FABRICTEMPORALSWITCHROUTE_H

#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::fabric {

/// One exact input-to-output-set signature presented to a Temporal switch.
/// The output inventory must be sorted, unique, and nonempty.
struct FabricTemporalSwitchRouteSignatureView final {
  FabricSwitchOccurrenceRef occurrence;
  FabricOrdinal input = 0;
  llvm::ArrayRef<FabricOrdinal> outputs;
};

/// One logical demand that must fit in a single resident route row. Re-entry
/// through one switch may contribute more than one input signature.
struct FabricTemporalSwitchRouteDemandView final {
  llvm::ArrayRef<FabricTemporalSwitchRouteSignatureView> signatures;
};

/// One selected route demand with the exact Physical Tag presented to the
/// switch table. Equal numeric tags in one occurrence address one resident
/// row; distinct tags necessarily occupy distinct rows.
struct FabricTemporalSwitchTaggedRouteDemandView final {
  FabricTemporalSwitchRouteDemandView route;
  llvm::APInt tag = llvm::APInt(1, 0);
};

/// The canonical occurrence-local resident-row projection. Demand ordinals
/// index the caller's input inventory. An incompatible row remains observable
/// for search-state accounting, but cannot be published or configured.
struct FabricTemporalSwitchPackedRouteRow final {
  FabricSwitchOccurrenceRef occurrence;
  llvm::APInt tag = llvm::APInt(1, 0);
  std::vector<std::uint64_t> demandOrdinals;
  bool compatible = true;
};

/// One search-state demand whose Physical Tag may still be unassigned. An
/// unassigned value has no configuration identity and cannot be persisted.
struct FabricTemporalSwitchCandidateRouteDemandView final {
  FabricTemporalSwitchRouteDemandView route;
  std::optional<llvm::APInt> tag;
};

/// Canonical row membership for a PnR candidate. Assigned rows retain their
/// exact Physical Tag; a tagless row is provisional search state only.
struct FabricTemporalSwitchCandidateRouteRow final {
  FabricSwitchOccurrenceRef occurrence;
  std::optional<llvm::APInt> tag;
  std::vector<std::uint64_t> demandOrdinals;
  bool compatible = true;
};

/// Returns whether two canonical signatures can occupy one resident row.
bool compatibleFabricTemporalSwitchRouteSignatures(
    const FabricTemporalSwitchRouteSignatureView &lhs,
    const FabricTemporalSwitchRouteSignatureView &rhs);

/// Validates the canonical shape and internal row compatibility of one demand.
llvm::Error validateFabricTemporalSwitchRouteDemand(
    const FabricTemporalSwitchRouteDemandView &demand);

/// Returns whether two individually valid demands can occupy one resident row
/// without adding an unrequested crosspoint, broadcast, or fan-in.
bool compatibleFabricTemporalSwitchRouteDemands(
    const FabricTemporalSwitchRouteDemandView &lhs,
    const FabricTemporalSwitchRouteDemandView &rhs);

/// Groups selected demands by exact `(occurrence, Physical Tag)`, orders rows
/// by occurrence and unsigned tag, and evaluates row compatibility through
/// the Fabric-owned relation above. This is the only resident-row membership
/// projection used by Mapping, PnR handshake, configuration, and execution.
llvm::Expected<std::vector<FabricTemporalSwitchPackedRouteRow>>
projectFabricTemporalSwitchRouteRows(
    llvm::ArrayRef<FabricTemporalSwitchTaggedRouteDemandView> demands);

/// Preserves every assigned `(occurrence, Physical Tag)` row, then places each
/// unassigned demand into the first compatible assigned or provisional row.
/// This search-only projection is the canonical lower bound while a PnR
/// candidate reports TagUnassigned. A final candidate must use the exact
/// tagged projection above.
llvm::Expected<std::vector<FabricTemporalSwitchCandidateRouteRow>>
projectFabricTemporalSwitchCandidateRouteRows(
    llvm::ArrayRef<FabricTemporalSwitchCandidateRouteDemandView> demands);

} // namespace loom::fabric

#endif // LOOM_FABRIC_IDENTITY_FABRICTEMPORALSWITCHROUTE_H
