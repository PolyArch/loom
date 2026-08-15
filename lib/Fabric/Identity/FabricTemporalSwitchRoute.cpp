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

llvm::Expected<std::vector<FabricTemporalSwitchCandidateRouteRow>>
loom::fabric::projectFabricTemporalSwitchCandidateRouteRows(
    llvm::ArrayRef<FabricTemporalSwitchCandidateRouteDemandView> demands) {
  struct PreparedRow final {
    FabricTemporalSwitchCandidateRouteRow row;
    std::vector<FabricTemporalSwitchRouteDemandView> routes;
  };
  std::map<FabricEntityId, std::vector<PreparedRow>> rows;
  std::map<FabricEntityId, unsigned> tagWidths;
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
    auto [width, inserted] =
        tagWidths.try_emplace(occurrence.id(), demand.tag->getBitWidth());
    if (!inserted && width->second != demand.tag->getBitWidth())
      return invalid("one switch occurrence has inconsistent tag widths");
    auto &occurrenceRows = rows[occurrence.id()];
    auto selected = llvm::find_if(occurrenceRows, [&](const PreparedRow &row) {
      return row.row.tag && compareUnsigned(*row.row.tag, *demand.tag) == 0;
    });
    if (selected == occurrenceRows.end()) {
      occurrenceRows.push_back({{occurrence, *demand.tag, {}, true}, {}});
      selected = std::prev(occurrenceRows.end());
    }
    selected->row.compatible &=
        llvm::all_of(selected->routes, [&](const auto &existing) {
          return compatibleFabricTemporalSwitchRouteDemands(existing,
                                                              demand.route);
        });
    selected->routes.push_back(demand.route);
    selected->row.demandOrdinals.push_back(ordinal);
  }
  for (auto &[occurrence, occurrenceRows] : rows) {
    (void)occurrence;
    llvm::sort(occurrenceRows, [](const auto &lhs, const auto &rhs) {
      return compareUnsigned(*lhs.row.tag, *rhs.row.tag) < 0;
    });
  }

  for (std::size_t ordinal = 0; ordinal != demands.size(); ++ordinal) {
    const FabricTemporalSwitchCandidateRouteDemandView &demand =
        demands[ordinal];
    if (demand.tag)
      continue;
    const FabricSwitchOccurrenceRef occurrence =
        demand.route.signatures.front().occurrence;
    auto &occurrenceRows = rows[occurrence.id()];
    auto selected = llvm::find_if(occurrenceRows, [&](const PreparedRow &row) {
      return row.row.compatible &&
             llvm::all_of(row.routes, [&](const auto &existing) {
               return compatibleFabricTemporalSwitchRouteDemands(
                   existing, demand.route);
             });
    });
    if (selected == occurrenceRows.end()) {
      occurrenceRows.push_back({{occurrence, std::nullopt, {}, true}, {}});
      selected = std::prev(occurrenceRows.end());
    }
    selected->routes.push_back(demand.route);
    selected->row.demandOrdinals.push_back(ordinal);
  }

  std::vector<FabricTemporalSwitchCandidateRouteRow> result;
  result.reserve(demands.size());
  for (auto &[occurrence, occurrenceRows] : rows) {
    (void)occurrence;
    for (PreparedRow &row : occurrenceRows)
      result.push_back(std::move(row.row));
  }
  return result;
}
