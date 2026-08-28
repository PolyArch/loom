#include "Fabric/Identity/FabricTemporalSwitchRoute.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <system_error>
#include <vector>

using namespace loom::fabric;

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "invalid Fabric Temporal switch route demand: " + message);
}

bool disjoint(llvm::ArrayRef<FabricOrdinal> lhs,
              llvm::ArrayRef<FabricOrdinal> rhs) {
  std::size_t left = 0;
  std::size_t right = 0;
  while (left != lhs.size() && right != rhs.size()) {
    if (lhs[left] == rhs[right])
      return false;
    if (lhs[left] < rhs[right])
      ++left;
    else
      ++right;
  }
  return true;
}

int compareUnsigned(const llvm::APInt &lhs, const llvm::APInt &rhs) {
  const unsigned width = std::max(lhs.getBitWidth(), rhs.getBitWidth());
  const llvm::APInt left = lhs.zext(width);
  const llvm::APInt right = rhs.zext(width);
  if (left.ult(right))
    return -1;
  if (right.ult(left))
    return 1;
  return 0;
}

} // namespace

bool loom::fabric::compatibleFabricTemporalSwitchRouteSignatures(
    const FabricTemporalSwitchRouteSignatureView &lhs,
    const FabricTemporalSwitchRouteSignatureView &rhs) {
  if (lhs.occurrence != rhs.occurrence)
    return false;
  if (lhs.input == rhs.input)
    return lhs.outputs == rhs.outputs;
  return disjoint(lhs.outputs, rhs.outputs);
}

llvm::Error loom::fabric::validateFabricTemporalSwitchRouteDemand(
    const FabricTemporalSwitchRouteDemandView &demand) {
  if (demand.signatures.empty())
    return invalid("demand has no input signature");
  FabricOrdinal previousInput = 0;
  bool first = true;
  for (const FabricTemporalSwitchRouteSignatureView &signature :
       demand.signatures) {
    if (signature.outputs.empty())
      return invalid("input signature has no selected output");
    if (!llvm::is_sorted(signature.outputs) ||
        std::adjacent_find(signature.outputs.begin(),
                           signature.outputs.end()) != signature.outputs.end())
      return invalid("input signature outputs are not sorted and unique");
    if (signature.occurrence != demand.signatures.front().occurrence)
      return invalid("one demand crosses switch occurrences");
    if (!first && signature.input <= previousInput)
      return invalid("input signatures are not ordered and unique");
    first = false;
    previousInput = signature.input;
  }
  for (std::size_t lhs = 0; lhs != demand.signatures.size(); ++lhs)
    for (std::size_t rhs = lhs + 1; rhs != demand.signatures.size(); ++rhs)
      if (!compatibleFabricTemporalSwitchRouteSignatures(
              demand.signatures[lhs], demand.signatures[rhs]))
        return invalid("one demand requires incompatible row signatures");
  return llvm::Error::success();
}

bool loom::fabric::compatibleFabricTemporalSwitchRouteDemands(
    const FabricTemporalSwitchRouteDemandView &lhs,
    const FabricTemporalSwitchRouteDemandView &rhs) {
  if (lhs.signatures.empty() || rhs.signatures.empty())
    return false;
  return llvm::all_of(lhs.signatures, [&](const auto &left) {
    return llvm::all_of(rhs.signatures, [&](const auto &right) {
      return compatibleFabricTemporalSwitchRouteSignatures(left, right);
    });
  });
}

llvm::Expected<std::vector<FabricTemporalSwitchPackedRouteRow>>
loom::fabric::projectFabricTemporalSwitchRouteRows(
    llvm::ArrayRef<FabricTemporalSwitchTaggedRouteDemandView> demands) {
  std::vector<FabricTemporalSwitchCandidateRouteDemandView> candidates;
  candidates.reserve(demands.size());
  for (const FabricTemporalSwitchTaggedRouteDemandView &demand : demands)
    candidates.push_back({demand.route, demand.tag});
  auto projected = projectFabricTemporalSwitchCandidateRouteRows(candidates);
  if (!projected)
    return projected.takeError();
  std::vector<FabricTemporalSwitchPackedRouteRow> result;
  result.reserve(projected->size());
  for (FabricTemporalSwitchCandidateRouteRow &row : *projected) {
    if (!row.tag)
      return invalid("exact projection produced an unassigned row");
    result.push_back({row.occurrence, std::move(*row.tag),
                      std::move(row.demandOrdinals), row.compatible});
  }
  return result;
}

namespace {

/// Flat row-grouping core shared by the materializing and counting
/// projections. Rows keep their demand members as index chains into one
/// node pool, so grouping allocates only three flat vectors regardless of
/// the demand count, and compatibility checks resolve members through the
/// caller's demand views without copying routes.
struct FlatRowState final {
  struct Row final {
    FabricSwitchOccurrenceRef occurrence;
    std::optional<llvm::APInt> tag;
    bool compatible = true;
    std::size_t headMember = SIZE_MAX;
    std::size_t tailMember = SIZE_MAX;
  };
  struct Occurrence final {
    FabricEntityId id;
    llvm::SmallVector<std::size_t, 4> rowOrder;
  };
  std::vector<Row> rows;
  /// Member chain node: demand ordinal plus next node index.
  std::vector<std::pair<std::size_t, std::size_t>> members;
  llvm::SmallVector<Occurrence, 8> occurrences;

  Occurrence &occurrenceFor(FabricEntityId id) {
    for (Occurrence &occurrence : occurrences)
      if (occurrence.id == id)
        return occurrence;
    occurrences.push_back({id, {}});
    return occurrences.back();
  }
  void appendMember(Row &row, std::size_t demandOrdinal) {
    members.push_back({demandOrdinal, SIZE_MAX});
    if (row.headMember == SIZE_MAX)
      row.headMember = members.size() - 1;
    else
      members[row.tailMember].second = members.size() - 1;
    row.tailMember = members.size() - 1;
  }
  template <typename Visit> bool allMembers(const Row &row, Visit visit) const {
    for (std::size_t node = row.headMember; node != SIZE_MAX;
         node = members[node].second)
      if (!visit(members[node].first))
        return false;
    return true;
  }
};

llvm::Expected<FlatRowState> prepareCandidateRouteRows(
    llvm::ArrayRef<FabricTemporalSwitchCandidateRouteDemandView> demands) {
  FlatRowState state;
  state.rows.reserve(demands.size());
  state.members.reserve(demands.size());
  llvm::SmallVector<std::pair<FabricEntityId, unsigned>, 8> tagWidths;
  for (std::size_t ordinal = 0; ordinal != demands.size(); ++ordinal) {
    const FabricTemporalSwitchCandidateRouteDemandView &demand =
        demands[ordinal];
    if (llvm::Error error =
            validateFabricTemporalSwitchRouteDemand(demand.route))
      return std::move(error);
    if (!demand.tag)
      continue;
    const FabricSwitchOccurrenceRef occurrence =
        demand.route.signatures.front().occurrence;
    bool widthKnown = false;
    for (const auto &width : tagWidths)
      if (width.first == occurrence.id()) {
        widthKnown = true;
        if (width.second != demand.tag->getBitWidth())
          return invalid("one switch occurrence has inconsistent tag widths");
        break;
      }
    if (!widthKnown)
      tagWidths.push_back({occurrence.id(), demand.tag->getBitWidth()});
    FlatRowState::Occurrence &group = state.occurrenceFor(occurrence.id());
    FlatRowState::Row *selected = nullptr;
    for (const std::size_t rowOrdinal : group.rowOrder) {
      FlatRowState::Row &row = state.rows[rowOrdinal];
      if (row.tag && compareUnsigned(*row.tag, *demand.tag) == 0) {
        selected = &row;
        break;
      }
    }
    if (!selected) {
      state.rows.push_back({occurrence, *demand.tag, true, SIZE_MAX, SIZE_MAX});
      group.rowOrder.push_back(state.rows.size() - 1);
      selected = &state.rows.back();
    }
    selected->compatible &=
        state.allMembers(*selected, [&](std::size_t member) {
          return compatibleFabricTemporalSwitchRouteDemands(
              demands[member].route, demand.route);
        });
    state.appendMember(*selected, ordinal);
  }
  for (FlatRowState::Occurrence &group : state.occurrences)
    llvm::sort(group.rowOrder, [&](std::size_t lhs, std::size_t rhs) {
      return compareUnsigned(*state.rows[lhs].tag, *state.rows[rhs].tag) < 0;
    });

  for (std::size_t ordinal = 0; ordinal != demands.size(); ++ordinal) {
    const FabricTemporalSwitchCandidateRouteDemandView &demand =
        demands[ordinal];
    if (demand.tag)
      continue;
    const FabricSwitchOccurrenceRef occurrence =
        demand.route.signatures.front().occurrence;
    FlatRowState::Occurrence &group = state.occurrenceFor(occurrence.id());
    FlatRowState::Row *selected = nullptr;
    for (const std::size_t rowOrdinal : group.rowOrder) {
      FlatRowState::Row &row = state.rows[rowOrdinal];
      if (row.compatible && state.allMembers(row, [&](std::size_t member) {
            return compatibleFabricTemporalSwitchRouteDemands(
                demands[member].route, demand.route);
          })) {
        selected = &row;
        break;
      }
    }
    if (!selected) {
      state.rows.push_back(
          {occurrence, std::nullopt, true, SIZE_MAX, SIZE_MAX});
      group.rowOrder.push_back(state.rows.size() - 1);
      selected = &state.rows.back();
    }
    state.appendMember(*selected, ordinal);
  }
  llvm::sort(state.occurrences, [](const FlatRowState::Occurrence &lhs,
                                   const FlatRowState::Occurrence &rhs) {
    return lhs.id < rhs.id;
  });
  return state;
}

} // namespace

llvm::Expected<std::vector<FabricTemporalSwitchCandidateRouteRow>>
loom::fabric::projectFabricTemporalSwitchCandidateRouteRows(
    llvm::ArrayRef<FabricTemporalSwitchCandidateRouteDemandView> demands) {
  auto state = prepareCandidateRouteRows(demands);
  if (!state)
    return state.takeError();
  std::vector<FabricTemporalSwitchCandidateRouteRow> result;
  result.reserve(state->rows.size());
  for (const FlatRowState::Occurrence &group : state->occurrences)
    for (const std::size_t rowOrdinal : group.rowOrder) {
      const FlatRowState::Row &row = state->rows[rowOrdinal];
      FabricTemporalSwitchCandidateRouteRow materialized;
      materialized.occurrence = row.occurrence;
      materialized.tag = row.tag;
      materialized.compatible = row.compatible;
      state->allMembers(row, [&](std::size_t member) {
        materialized.demandOrdinals.push_back(member);
        return true;
      });
      result.push_back(std::move(materialized));
    }
  return result;
}

llvm::Expected<std::uint64_t>
loom::fabric::projectFabricTemporalSwitchCandidateRouteRowCount(
    llvm::ArrayRef<FabricTemporalSwitchCandidateRouteDemandView> demands) {
  auto state = prepareCandidateRouteRows(demands);
  if (!state)
    return state.takeError();
  return static_cast<std::uint64_t>(state->rows.size());
}
