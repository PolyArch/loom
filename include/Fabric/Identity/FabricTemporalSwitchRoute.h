#ifndef LOOM_FABRIC_IDENTITY_FABRICTEMPORALSWITCHROUTE_H
#define LOOM_FABRIC_IDENTITY_FABRICTEMPORALSWITCHROUTE_H

#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <memory>
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
/// The assigned tag is borrowed for the duration of one projection; numeric
/// value, not APInt storage width, is semantic.
struct FabricTemporalSwitchCandidateRouteDemandView final {
  FabricTemporalSwitchRouteDemandView route;
  const llvm::APInt *tag = nullptr;
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

/// Number of resident rows the candidate projection above would produce,
/// without materializing the rows. Row membership and greedy placement are
/// shared with the full projection, so the count is exactly its size.
/// Flat row membership of the candidate projection in canonical row order:
/// row r's member demand ordinals are demandOrdinals[rowOffsets[r]] through
/// demandOrdinals[rowOffsets[r + 1]]. Refilling reuses retained capacity.
struct FabricTemporalSwitchRouteRowMemberSpans final {
  std::vector<std::uint64_t> rowOffsets;
  std::vector<std::uint64_t> demandOrdinals;
};

/// Reusable storage for the canonical candidate-row projection. It carries no
/// route demand, row membership, or Fabric identity between calls.
class FabricTemporalSwitchCandidateRouteProjectionScratch final {
public:
  FabricTemporalSwitchCandidateRouteProjectionScratch();
  FabricTemporalSwitchCandidateRouteProjectionScratch(
      const FabricTemporalSwitchCandidateRouteProjectionScratch &) = delete;
  FabricTemporalSwitchCandidateRouteProjectionScratch &operator=(
      const FabricTemporalSwitchCandidateRouteProjectionScratch &) = delete;
  ~FabricTemporalSwitchCandidateRouteProjectionScratch();

  /// Retains capacity for at least this many demands. Preparation carries no
  /// semantic state and may be repeated with larger bounds.
  void prepare(std::size_t demandCapacity);
  std::size_t retainedStorageBytes() const;

  /// Opaque reusable buffers; only canonical projection functions can access
  /// their contents.
  struct Storage;

private:
  std::unique_ptr<Storage> storage_;

  friend llvm::Expected<std::vector<FabricTemporalSwitchCandidateRouteRow>>
  projectFabricTemporalSwitchCandidateRouteRows(
      llvm::ArrayRef<FabricTemporalSwitchCandidateRouteDemandView> demands);
  friend llvm::Error projectFabricTemporalSwitchCandidateRouteRowMemberSpans(
      llvm::ArrayRef<FabricTemporalSwitchCandidateRouteDemandView> demands,
      FabricTemporalSwitchRouteRowMemberSpans &result,
      FabricTemporalSwitchCandidateRouteProjectionScratch &scratch);
  friend llvm::Expected<std::uint64_t>
  projectFabricTemporalSwitchCandidateRouteRowCount(
      llvm::ArrayRef<FabricTemporalSwitchCandidateRouteDemandView> demands,
      FabricTemporalSwitchCandidateRouteProjectionScratch &scratch);
};

llvm::Error projectFabricTemporalSwitchCandidateRouteRowMemberSpans(
    llvm::ArrayRef<FabricTemporalSwitchCandidateRouteDemandView> demands,
    FabricTemporalSwitchRouteRowMemberSpans &result);
llvm::Error projectFabricTemporalSwitchCandidateRouteRowMemberSpans(
    llvm::ArrayRef<FabricTemporalSwitchCandidateRouteDemandView> demands,
    FabricTemporalSwitchRouteRowMemberSpans &result,
    FabricTemporalSwitchCandidateRouteProjectionScratch &scratch);

llvm::Expected<std::uint64_t> projectFabricTemporalSwitchCandidateRouteRowCount(
    llvm::ArrayRef<FabricTemporalSwitchCandidateRouteDemandView> demands);
llvm::Expected<std::uint64_t> projectFabricTemporalSwitchCandidateRouteRowCount(
    llvm::ArrayRef<FabricTemporalSwitchCandidateRouteDemandView> demands,
    FabricTemporalSwitchCandidateRouteProjectionScratch &scratch);

} // namespace loom::fabric

#endif // LOOM_FABRIC_IDENTITY_FABRICTEMPORALSWITCHROUTE_H
