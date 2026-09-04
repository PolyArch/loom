#include "Fabric/Identity/FabricTemporalSwitchRoute.h"

#include "Fabric/IR/PhysicalTag.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <numeric>
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
  return ::fabric::comparePhysicalTagValues(lhs, rhs);
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
    candidates.push_back({demand.route, &demand.tag});
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

/// Reusable flat row-grouping storage shared by the materializing and
/// counting projections. Rows borrow tags from the caller for the duration of
/// one projection. Membership uses index chains into one node pool, avoiding
/// per-occurrence and per-row allocation.
struct loom::fabric::FabricTemporalSwitchCandidateRouteProjectionScratch::
    Storage final {
  struct Row final {
    FabricSwitchOccurrenceRef occurrence;
    const llvm::APInt *tag = nullptr;
    bool compatible = true;
    std::size_t headMember = SIZE_MAX;
    std::size_t tailMember = SIZE_MAX;
  };
  std::vector<Row> rows;
  /// Member chain node: demand ordinal plus next node index.
  std::vector<std::pair<std::size_t, std::size_t>> members;
  std::vector<std::size_t> rowOrder;

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

namespace {

using CandidateRouteProjectionStorage =
    FabricTemporalSwitchCandidateRouteProjectionScratch::Storage;

llvm::Error prepareCandidateRouteRows(
    llvm::ArrayRef<FabricTemporalSwitchCandidateRouteDemandView> demands,
    CandidateRouteProjectionStorage &state) {
  state.rows.clear();
  state.members.clear();
  state.rowOrder.clear();
  state.rows.reserve(demands.size());
  state.members.reserve(demands.size());
  state.rowOrder.reserve(demands.size());
  for (std::size_t ordinal = 0; ordinal != demands.size(); ++ordinal) {
    const FabricTemporalSwitchCandidateRouteDemandView &demand =
        demands[ordinal];
    if (llvm::Error error =
            validateFabricTemporalSwitchRouteDemand(demand.route))
      return error;
    if (!demand.tag)
      continue;
    const FabricSwitchOccurrenceRef occurrence =
        demand.route.signatures.front().occurrence;
    CandidateRouteProjectionStorage::Row *selected = nullptr;
    for (CandidateRouteProjectionStorage::Row &row : state.rows) {
      if (row.occurrence == occurrence && row.tag &&
          compareUnsigned(*row.tag, *demand.tag) == 0) {
        selected = &row;
        break;
      }
    }
    if (!selected) {
      state.rows.push_back({occurrence, demand.tag, true, SIZE_MAX, SIZE_MAX});
      selected = &state.rows.back();
    }
    selected->compatible &=
        state.allMembers(*selected, [&](std::size_t member) {
          return compatibleFabricTemporalSwitchRouteDemands(
              demands[member].route, demand.route);
        });
    state.appendMember(*selected, ordinal);
  }
  state.rowOrder.resize(state.rows.size());
  std::iota(state.rowOrder.begin(), state.rowOrder.end(), 0);
  std::sort(state.rowOrder.begin(), state.rowOrder.end(),
            [&](std::size_t lhs, std::size_t rhs) {
              const auto &left = state.rows[lhs];
              const auto &right = state.rows[rhs];
              if (left.occurrence.id() != right.occurrence.id())
                return left.occurrence.id() < right.occurrence.id();
              return compareUnsigned(*left.tag, *right.tag) < 0;
            });

  for (std::size_t ordinal = 0; ordinal != demands.size(); ++ordinal) {
    const FabricTemporalSwitchCandidateRouteDemandView &demand =
        demands[ordinal];
    if (demand.tag)
      continue;
    const FabricSwitchOccurrenceRef occurrence =
        demand.route.signatures.front().occurrence;
    CandidateRouteProjectionStorage::Row *selected = nullptr;
    for (const std::size_t rowOrdinal : state.rowOrder) {
      CandidateRouteProjectionStorage::Row &row = state.rows[rowOrdinal];
      if (row.occurrence != occurrence)
        continue;
      if (row.compatible && state.allMembers(row, [&](std::size_t member) {
            return compatibleFabricTemporalSwitchRouteDemands(
                demands[member].route, demand.route);
          })) {
        selected = &row;
        break;
      }
    }
    if (!selected) {
      state.rows.push_back({occurrence, nullptr, true, SIZE_MAX, SIZE_MAX});
      const std::size_t rowOrdinal = state.rows.size() - 1;
      const auto position =
          llvm::upper_bound(state.rowOrder, occurrence.id(),
                            [&](FabricEntityId id, std::size_t existing) {
                              return id < state.rows[existing].occurrence.id();
                            });
      state.rowOrder.insert(position, rowOrdinal);
      selected = &state.rows.back();
    }
    state.appendMember(*selected, ordinal);
  }
  return llvm::Error::success();
}

} // namespace

FabricTemporalSwitchCandidateRouteProjectionScratch::
    FabricTemporalSwitchCandidateRouteProjectionScratch()
    : storage_(std::make_unique<Storage>()) {}

FabricTemporalSwitchCandidateRouteProjectionScratch::
    ~FabricTemporalSwitchCandidateRouteProjectionScratch() = default;

void FabricTemporalSwitchCandidateRouteProjectionScratch::prepare(
    std::size_t demandCapacity) {
  storage_->rows.reserve(demandCapacity);
  storage_->members.reserve(demandCapacity);
  storage_->rowOrder.reserve(demandCapacity);
}

std::size_t
FabricTemporalSwitchCandidateRouteProjectionScratch::retainedStorageBytes()
    const {
  return storage_->rows.capacity() * sizeof(Storage::Row) +
         storage_->members.capacity() *
             sizeof(std::pair<std::size_t, std::size_t>) +
         storage_->rowOrder.capacity() * sizeof(std::size_t);
}

llvm::Expected<std::vector<FabricTemporalSwitchCandidateRouteRow>>
loom::fabric::projectFabricTemporalSwitchCandidateRouteRows(
    llvm::ArrayRef<FabricTemporalSwitchCandidateRouteDemandView> demands) {
  FabricTemporalSwitchCandidateRouteProjectionScratch scratch;
  CandidateRouteProjectionStorage &state = *scratch.storage_;
  if (llvm::Error error = prepareCandidateRouteRows(demands, state))
    return std::move(error);
  std::vector<FabricTemporalSwitchCandidateRouteRow> result;
  result.reserve(state.rows.size());
  for (const std::size_t rowOrdinal : state.rowOrder) {
    const CandidateRouteProjectionStorage::Row &row = state.rows[rowOrdinal];
    FabricTemporalSwitchCandidateRouteRow materialized;
    materialized.occurrence = row.occurrence;
    if (row.tag)
      materialized.tag = ::fabric::canonicalPhysicalTagValue(*row.tag);
    materialized.compatible = row.compatible;
    state.allMembers(row, [&](std::size_t member) {
      materialized.demandOrdinals.push_back(member);
      return true;
    });
    result.push_back(std::move(materialized));
  }
  return result;
}

llvm::Error
loom::fabric::projectFabricTemporalSwitchCandidateRouteRowMemberSpans(
    llvm::ArrayRef<FabricTemporalSwitchCandidateRouteDemandView> demands,
    FabricTemporalSwitchRouteRowMemberSpans &result) {
  FabricTemporalSwitchCandidateRouteProjectionScratch scratch;
  return projectFabricTemporalSwitchCandidateRouteRowMemberSpans(
      demands, result, scratch);
}

llvm::Error
loom::fabric::projectFabricTemporalSwitchCandidateRouteRowMemberSpans(
    llvm::ArrayRef<FabricTemporalSwitchCandidateRouteDemandView> demands,
    FabricTemporalSwitchRouteRowMemberSpans &result,
    FabricTemporalSwitchCandidateRouteProjectionScratch &scratch) {
  CandidateRouteProjectionStorage &state = *scratch.storage_;
  if (llvm::Error error = prepareCandidateRouteRows(demands, state))
    return error;
  result.rowOffsets.clear();
  result.demandOrdinals.clear();
  result.rowOffsets.reserve(state.rows.size() + 1);
  result.demandOrdinals.reserve(state.members.size());
  for (const std::size_t rowOrdinal : state.rowOrder) {
    result.rowOffsets.push_back(result.demandOrdinals.size());
    state.allMembers(state.rows[rowOrdinal], [&](std::size_t member) {
      result.demandOrdinals.push_back(member);
      return true;
    });
  }
  result.rowOffsets.push_back(result.demandOrdinals.size());
  return llvm::Error::success();
}

llvm::Expected<std::uint64_t>
loom::fabric::projectFabricTemporalSwitchCandidateRouteRowCount(
    llvm::ArrayRef<FabricTemporalSwitchCandidateRouteDemandView> demands) {
  FabricTemporalSwitchCandidateRouteProjectionScratch scratch;
  return projectFabricTemporalSwitchCandidateRouteRowCount(demands, scratch);
}

llvm::Expected<std::uint64_t>
loom::fabric::projectFabricTemporalSwitchCandidateRouteRowCount(
    llvm::ArrayRef<FabricTemporalSwitchCandidateRouteDemandView> demands,
    FabricTemporalSwitchCandidateRouteProjectionScratch &scratch) {
  CandidateRouteProjectionStorage &state = *scratch.storage_;
  if (llvm::Error error = prepareCandidateRouteRows(demands, state))
    return std::move(error);
  return static_cast<std::uint64_t>(state.rows.size());
}
