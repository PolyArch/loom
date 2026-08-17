#include "TechMappingCandidate.h"

#include "Common/MappingDebugLog.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <utility>
#include <vector>

namespace loom::mapping::detail {
namespace {

using ActorMask = std::vector<std::uint64_t>;

std::uint64_t saturatingAdd(std::uint64_t lhs, std::uint64_t rhs) {
  return rhs > std::numeric_limits<std::uint64_t>::max() - lhs
             ? std::numeric_limits<std::uint64_t>::max()
             : lhs + rhs;
}

std::uint64_t rowSupplyBreadth(const TechMatchRow &row) {
  if (std::holds_alternative<TechComputeRealizationView>(row.realization))
    return row.computeContextValues.size();
  return row.memoryOccurrenceDemand
             ? row.memoryOccurrenceDemand->occurrences.size()
             : 0;
}

std::uint64_t coverSupplyBreadth(llvm::ArrayRef<const TechMatchRow *> cover) {
  std::uint64_t result = 0;
  for (const TechMatchRow *row : cover)
    result = saturatingAdd(result, rowSupplyBreadth(*row));
  return result;
}

SpatialComputeContextSupplyAnalysis
analyzeComputeContextSupply(llvm::ArrayRef<const TechMatchRow *> rows,
                            const TechMatchDomain &domain,
                            TechMappingGenerationAccounting &accounting) {
  std::vector<std::vector<std::size_t>> domains;
  for (const TechMatchRow *row : rows)
    if (std::holds_alternative<TechComputeRealizationView>(row->realization))
      domains.push_back(row->computeContextValues);
  ++accounting.computeContextMatchingChecks;
  SpatialComputeContextSupplyAnalysis analysis =
      llvm::cantFail(analyzeSpatialComputeContextSupply(
          domains, domain.computeContextValueCount));
  accounting.computeContextMatchingWork = saturatingAdd(
      accounting.computeContextMatchingWork, analysis.deterministicWork);
  accounting.computeContextRejectedChecks += !analysis.admissible();
  return analysis;
}

enum class MemorySupplyCheckScope : std::uint8_t {
  PartialCover,
  FullCover,
};

SpatialMemoryOccurrenceSupplyAnalysis
analyzeMemoryOccurrenceSupply(llvm::ArrayRef<const TechMatchRow *> rows,
                              TechMappingGenerationAccounting &accounting,
                              MemorySupplyCheckScope scope) {
  std::vector<const SpatialMemoryOccurrenceDemandView *> demands;
  for (const TechMatchRow *row : rows) {
    if (!std::holds_alternative<TechMemoryRealizationView>(row->realization))
      continue;
    assert(row->memoryOccurrenceDemand &&
           "memory Tech row has no occurrence-demand projection");
    demands.push_back(&*row->memoryOccurrenceDemand);
  }
  ++accounting.memorySupplyChecks;
  if (scope == MemorySupplyCheckScope::PartialCover)
    ++accounting.memorySupplyPartialChecks;
  else
    ++accounting.memorySupplyFullChecks;
  SpatialMemoryOccurrenceSupplyAnalysis analysis =
      llvm::cantFail(analyzeSpatialMemoryOccurrenceSupply(demands));
  accounting.memorySupplySearchWork = saturatingAdd(
      accounting.memorySupplySearchWork, analysis.deterministicWork);
  if (!analysis.admissible()) {
    ++accounting.memorySupplyRejectedChecks;
    switch (analysis.failure) {
    case SpatialMemoryOccurrenceSupplyFailureKind::None:
      llvm_unreachable("an admissible memory supply was classified rejected");
    case SpatialMemoryOccurrenceSupplyFailureKind::EmptyOccurrenceDomain:
      ++accounting.memorySupplyEmptyDomainRejections;
      break;
    case SpatialMemoryOccurrenceSupplyFailureKind::ExclusiveResourceDeficit:
      ++accounting.memorySupplyExclusiveResourceRejections;
      assert(analysis.failingResourceKind &&
             "exclusive-resource failure has no resource kind");
      switch (*analysis.failingResourceKind) {
      case SpatialMemoryExclusiveResourceKind::SpatialOperationPort:
        ++accounting.memorySupplySpatialPortRejections;
        break;
      case SpatialMemoryExclusiveResourceKind::TemporalExternalIngress:
        ++accounting.memorySupplyTemporalIngressRejections;
        break;
      case SpatialMemoryExclusiveResourceKind::InternalConnection:
        ++accounting.memorySupplyInternalConnectionRejections;
        break;
      }
      break;
    case SpatialMemoryOccurrenceSupplyFailureKind::ResidentCapacityDeficit:
      ++accounting.memorySupplyResidentCapacityRejections;
      break;
    case SpatialMemoryOccurrenceSupplyFailureKind::JointAssignmentInfeasible:
      ++accounting.memorySupplyJointAssignmentRejections;
      break;
    }
  }
  return analysis;
}

void emitComputeContextRejection(
    std::uint64_t coverOrdinal, llvm::ArrayRef<const TechMatchRow *> rows,
    const SpatialComputeContextSupplyAnalysis &analysis) {
  mapping_debug::emit(
      mapping_debug::Level::Detail, mapping_debug::Stage::TechMapping,
      mapping_debug::Event::MappingFailure, [&](llvm::json::Object &fields) {
        std::vector<const TechMatchRow *> computeRows;
        std::map<std::uint64_t, std::uint64_t> widths;
        for (const TechMatchRow *row : rows) {
          if (!std::holds_alternative<TechComputeRealizationView>(
                  row->realization))
            continue;
          computeRows.push_back(row);
          ++widths[row->computeContextValues.size()];
        }
        std::map<std::uint64_t, std::uint64_t> hallWidths;
        for (const std::uint64_t demand : analysis.hallDemands)
          ++hallWidths[computeRows[demand]->computeContextValues.size()];
        const auto histogram = [](const auto &counts) {
          llvm::json::Array result;
          for (const auto &[width, count] : counts) {
            llvm::json::Object bucket;
            bucket["domain_width"] = width;
            bucket["demand_count"] = count;
            result.push_back(std::move(bucket));
          }
          return result;
        };
        fields["failure_scope"] = "tech_cover_compute_context_supply";
        fields["closure_status"] = "proven_infeasible";
        fields["cover_ordinal"] = coverOrdinal;
        fields["row_count"] = rows.size();
        fields["compute_demand_count"] = analysis.demandCount;
        fields["compute_context_value_count"] = analysis.valueCount;
        fields["compute_context_edge_count"] = analysis.edgeCount;
        fields["compute_context_maximum_matching"] = analysis.maximumMatching;
        fields["compute_hall_demand_count"] = analysis.hallDemands.size();
        fields["compute_hall_context_value_count"] = analysis.hallValueCount;
        fields["domain_width_histogram"] = histogram(widths);
        fields["hall_domain_width_histogram"] = histogram(hallWidths);
      });
}

void emitMemorySupplyRejection(
    std::uint64_t coverOrdinal, llvm::ArrayRef<const TechMatchRow *> rows,
    const SpatialMemoryOccurrenceSupplyAnalysis &analysis) {
  mapping_debug::emit(
      mapping_debug::Level::Detail, mapping_debug::Stage::TechMapping,
      mapping_debug::Event::MappingFailure, [&](llvm::json::Object &fields) {
        fields["failure_scope"] = "tech_cover_memory_occurrence_supply";
        fields["closure_status"] = "proven_infeasible";
        fields["cover_ordinal"] = coverOrdinal;
        fields["row_count"] = rows.size();
        fields["failure_kind"] =
            spatialMemoryOccurrenceSupplyFailureKindSpelling(analysis.failure);
        fields["memory_demand_count"] = analysis.demandCount;
        fields["memory_occurrence_value_count"] = analysis.occurrenceValueCount;
        fields["memory_occurrence_choice_count"] =
            analysis.occurrenceChoiceCount;
        fields["memory_exclusive_relation_count"] =
            analysis.exclusiveRelationCount;
        fields["memory_assignment_attempts"] = analysis.assignmentAttempts;
        fields["failing_demand_count"] = analysis.failingDemandCount;
        fields["failing_occurrence_count"] = analysis.failingOccurrenceCount;
        fields["failing_resident_demand"] = analysis.failingResidentDemand;
        fields["failing_resident_capacity"] = analysis.failingResidentCapacity;
        if (analysis.failingResourceKind)
          fields["failing_resource_kind"] =
              spatialMemoryExclusiveResourceKindSpelling(
                  *analysis.failingResourceKind);
      });
}

bool rootSupplyAdmissible(llvm::ArrayRef<const TechMatchRow *> rows,
                          const TechMatchDomain &domain,
                          TechMappingGenerationAccounting &accounting) {
  const std::size_t computeCount = llvm::count_if(rows, [](const auto *row) {
    return std::holds_alternative<TechComputeRealizationView>(row->realization);
  });
  if (computeCount > 1 &&
      !analyzeComputeContextSupply(rows, domain, accounting).admissible())
    return false;
  const std::size_t memoryCount = llvm::count_if(rows, [](const auto *row) {
    return std::holds_alternative<TechMemoryRealizationView>(row->realization);
  });
  return memoryCount <= 1 ||
         analyzeMemoryOccurrenceSupply(rows, accounting,
                                       MemorySupplyCheckScope::PartialCover)
             .admissible();
}

std::size_t actorMaskWordCount(std::size_t actorCount) {
  return (actorCount + 63) / 64;
}

void setActor(ActorMask &mask, std::size_t actor) {
  mask[actor / 64] |= std::uint64_t{1} << (actor % 64);
}

bool containsActor(const ActorMask &mask, std::size_t actor) {
  return (mask[actor / 64] & (std::uint64_t{1} << (actor % 64))) != 0;
}

bool masksIntersect(const ActorMask &lhs, const ActorMask &rhs) {
  for (std::size_t word = 0; word < lhs.size(); ++word)
    if ((lhs[word] & rhs[word]) != 0)
      return true;
  return false;
}

void mergeMask(ActorMask &destination, const ActorMask &source) {
  for (std::size_t word = 0; word < destination.size(); ++word)
    destination[word] |= source[word];
}

bool coversMask(const ActorMask &covered, const ActorMask &required) {
  for (std::size_t word = 0; word < covered.size(); ++word)
    if ((covered[word] & required[word]) != required[word])
      return false;
  return true;
}

struct IncidenceComponent final {
  std::vector<std::size_t> actors;
  std::vector<std::size_t> rows;
};

struct ComponentSearchState final {
  ActorMask covered;
  std::vector<std::size_t> selectedRows;
  std::vector<std::size_t> lowerBound;
  std::uint64_t lowerBoundSupplyBreadth = 0;
};

// `lowerBound` is a lower bound in the complete-cover order. Its length is an
// admissible realization-count bound. At that length, its row sequence is the
// most root-flexible unconstrained completion of the selected rows. Canonical
// row order is the final deterministic tie-break.
struct ComponentStateGreater final {
  bool operator()(const ComponentSearchState &lhs,
                  const ComponentSearchState &rhs) const {
    if (lhs.lowerBound.size() != rhs.lowerBound.size())
      return lhs.lowerBound.size() > rhs.lowerBound.size();
    if (lhs.lowerBoundSupplyBreadth != rhs.lowerBoundSupplyBreadth)
      return lhs.lowerBoundSupplyBreadth < rhs.lowerBoundSupplyBreadth;
    if (lhs.lowerBound != rhs.lowerBound)
      return lhs.lowerBound > rhs.lowerBound;
    return lhs.selectedRows > rhs.selectedRows;
  }
};

enum class ComponentAdvanceKind : std::uint8_t {
  Cover,
  Exhausted,
  Interrupted,
};

struct ComponentAdvance final {
  ComponentAdvanceKind kind;
  std::vector<const TechMatchRow *> cover;
};

std::vector<IncidenceComponent> componentsOf(const TechMatchDomain &domain) {
  std::vector<std::vector<std::size_t>> rowsByActor(domain.actors.size());
  for (auto [rowIndex, row] : llvm::enumerate(domain.rows))
    for (std::size_t actor : row.actorSlots)
      rowsByActor[actor].push_back(rowIndex);

  std::vector<bool> visitedActor(domain.actors.size(), false);
  std::vector<bool> visitedRow(domain.rows.size(), false);
  std::vector<IncidenceComponent> components;
  for (std::size_t seed = 0; seed < domain.actors.size(); ++seed) {
    if (visitedActor[seed])
      continue;
    IncidenceComponent component;
    component.actors.push_back(seed);
    visitedActor[seed] = true;
    for (std::size_t cursor = 0; cursor < component.actors.size(); ++cursor) {
      const std::size_t actor = component.actors[cursor];
      for (std::size_t row : rowsByActor[actor]) {
        if (!visitedRow[row]) {
          visitedRow[row] = true;
          component.rows.push_back(row);
        }
        for (std::size_t adjacent : domain.rows[row].actorSlots)
          if (!visitedActor[adjacent]) {
            visitedActor[adjacent] = true;
            component.actors.push_back(adjacent);
          }
      }
    }
    llvm::sort(component.actors);
    llvm::sort(component.rows);
    components.push_back(std::move(component));
  }
  return components;
}

class ComponentCoverCursor final {
public:
  ComponentCoverCursor(const TechMatchDomain &domain,
                       const IncidenceComponent &component,
                       llvm::ArrayRef<std::vector<std::size_t>> rowsByActor,
                       llvm::ArrayRef<ActorMask> rowMasks,
                       llvm::ArrayRef<std::size_t> actorLocalSlots,
                       const ResolvedTechMappingConfigView &config,
                       TechMappingGenerationAccounting &accounting,
                       ExecutionControlView executionControl)
      : domain_(domain), component_(component), rowsByActor_(rowsByActor),
        rowMasks_(rowMasks), actorLocalSlots_(actorLocalSlots),
        componentMask_(actorMaskWordCount(component.actors.size()), 0),
        config_(config), accounting_(accounting),
        executionControl_(executionControl) {
    for (std::size_t actor = 0; actor < component_.actors.size(); ++actor)
      setActor(componentMask_, actor);
  }

  ComponentAdvance next() {
    if (executionControl_.stopRequested())
      return {ComponentAdvanceKind::Interrupted, {}};
    if (exhausted_)
      return {ComponentAdvanceKind::Exhausted, {}};
    if (!initialized_)
      initialize();
    if (executionControl_.stopRequested())
      return {ComponentAdvanceKind::Interrupted, {}};
    if (exhausted_)
      return {ComponentAdvanceKind::Exhausted, {}};
    if (sealed_)
      return nextSealed();

    while (!pending_.empty()) {
      if (executionControl_.stopRequested())
        return {ComponentAdvanceKind::Interrupted, {}};
      ComponentSearchState state = pending_.top();
      const bool complete = coversMask(state.covered, componentMask_);
      if (complete) {
        pending_.pop();
        std::vector<const TechMatchRow *> cover;
        cover.reserve(state.selectedRows.size());
        for (std::size_t row : state.selectedRows)
          cover.push_back(&domain_.rows[row]);
        return {ComponentAdvanceKind::Cover, std::move(cover)};
      }
      pending_.pop();

      std::vector<std::size_t> selectedOptions;
      std::size_t smallestDomain = std::numeric_limits<std::size_t>::max();
      for (std::size_t actor : component_.actors) {
        if (covered(state, actor))
          continue;
        std::vector<std::size_t> compatible =
            compatibleRows(state, actor, smallestDomain);
        if (compatible.size() < smallestDomain) {
          smallestDomain = compatible.size();
          selectedOptions = std::move(compatible);
        }
      }

      for (std::size_t row : selectedOptions) {
        if (executionControl_.stopRequested())
          return {ComponentAdvanceKind::Interrupted, {}};
        if (!consumeExpansion()) {
          seal();
          return nextSealed();
        }
        ComponentSearchState child = state;
        addRow(child, row);
        const PropagationResult propagation = propagate(child);
        if (propagation == PropagationResult::Interrupted)
          return {ComponentAdvanceKind::Interrupted, {}};
        if (propagation == PropagationResult::LimitReached) {
          seal();
          return nextSealed();
        }
        if (propagation == PropagationResult::Viable &&
            rootSupplyAdmissible(child) && setLowerBound(child))
          pending_.push(std::move(child));
      }
    }
    exhausted_ = true;
    return {ComponentAdvanceKind::Exhausted, {}};
  }

  bool truncated() const { return truncated_; }

private:
  enum class PropagationResult : std::uint8_t {
    Viable,
    Infeasible,
    LimitReached,
    Interrupted,
  };

  std::vector<std::size_t> compatibleRows(const ComponentSearchState &state,
                                          std::size_t actor,
                                          std::size_t resultLimit) const {
    std::vector<std::size_t> compatible;
    for (std::size_t row : rowsByActor_[actor]) {
      if (!masksIntersect(state.covered, rowMasks_[row])) {
        compatible.push_back(row);
        if (compatible.size() == resultLimit)
          break;
      }
    }
    return compatible;
  }

  bool covered(const ComponentSearchState &state, std::size_t actor) const {
    return containsActor(state.covered, actorLocalSlots_[actor]);
  }

  void addRow(ComponentSearchState &state, std::size_t row) const {
    state.selectedRows.insert(llvm::lower_bound(state.selectedRows, row), row);
    mergeMask(state.covered, rowMasks_[row]);
  }

  bool rootSupplyAdmissible(const ComponentSearchState &state) const {
    std::vector<const TechMatchRow *> rows;
    rows.reserve(state.selectedRows.size());
    for (const std::size_t row : state.selectedRows)
      rows.push_back(&domain_.rows[row]);
    return detail::rootSupplyAdmissible(rows, domain_, accounting_);
  }

  bool consumeExpansion() {
    if (accounting_.partialCoverExpansions >=
        config_.partialCoverExpansionLimit()) {
      truncated_ = true;
      return false;
    }
    ++accounting_.partialCoverExpansions;
    return true;
  }

  PropagationResult propagate(ComponentSearchState &state) {
    while (true) {
      if (executionControl_.stopRequested())
        return PropagationResult::Interrupted;
      std::optional<std::size_t> forcedRow;
      for (std::size_t actor : component_.actors) {
        if (covered(state, actor))
          continue;
        const std::vector<std::size_t> compatible =
            compatibleRows(state, actor, 2);
        if (compatible.empty())
          return PropagationResult::Infeasible;
        if (compatible.size() == 1) {
          forcedRow = compatible.front();
          break;
        }
      }
      if (!forcedRow)
        return PropagationResult::Viable;
      if (!consumeExpansion())
        return PropagationResult::LimitReached;
      addRow(state, *forcedRow);
    }
  }

  bool setLowerBound(ComponentSearchState &state) const {
    state.lowerBound = state.selectedRows;
    state.lowerBoundSupplyBreadth = 0;
    for (const std::size_t row : state.selectedRows)
      state.lowerBoundSupplyBreadth = saturatingAdd(
          state.lowerBoundSupplyBreadth, rowSupplyBreadth(domain_.rows[row]));
    std::size_t uncoveredActorCount = 0;
    for (std::size_t actor : component_.actors)
      uncoveredActorCount += !covered(state, actor);
    if (uncoveredActorCount == 0)
      return true;

    std::vector<std::size_t> availableRows;
    std::size_t maximumNewActorsPerRow = 0;
    for (std::size_t row : component_.rows) {
      if (masksIntersect(state.covered, rowMasks_[row]))
        continue;
      availableRows.push_back(row);
      maximumNewActorsPerRow =
          std::max(maximumNewActorsPerRow, domain_.rows[row].actorSlots.size());
    }
    if (maximumNewActorsPerRow == 0)
      return false;

    const std::size_t additionalRowLowerBound =
        (uncoveredActorCount + maximumNewActorsPerRow - 1) /
        maximumNewActorsPerRow;
    if (availableRows.size() < additionalRowLowerBound)
      return false;
    llvm::sort(availableRows, [&](std::size_t lhs, std::size_t rhs) {
      const std::uint64_t lhsBreadth = rowSupplyBreadth(domain_.rows[lhs]);
      const std::uint64_t rhsBreadth = rowSupplyBreadth(domain_.rows[rhs]);
      if (lhsBreadth != rhsBreadth)
        return lhsBreadth > rhsBreadth;
      return domain_.rows[lhs].key < domain_.rows[rhs].key;
    });
    for (std::size_t ordinal = 0; ordinal != additionalRowLowerBound; ++ordinal)
      state.lowerBoundSupplyBreadth =
          saturatingAdd(state.lowerBoundSupplyBreadth,
                        rowSupplyBreadth(domain_.rows[availableRows[ordinal]]));
    state.lowerBound.insert(state.lowerBound.end(), availableRows.begin(),
                            availableRows.begin() + additionalRowLowerBound);
    llvm::sort(state.lowerBound);
    return true;
  }

  void initialize() {
    initialized_ = true;
    ComponentSearchState initial{
        ActorMask(actorMaskWordCount(component_.actors.size()), 0), {}, {}, 0};
    const PropagationResult propagation = propagate(initial);
    if (propagation == PropagationResult::Interrupted)
      return;
    if (propagation == PropagationResult::LimitReached) {
      seal();
      return;
    }
    if (propagation == PropagationResult::Infeasible ||
        !rootSupplyAdmissible(initial) || !setLowerBound(initial)) {
      exhausted_ = true;
      return;
    }
    pending_.push(std::move(initial));
  }

  void seal() {
    if (sealed_)
      return;
    sealed_ = true;
    truncated_ = true;
    while (!pending_.empty()) {
      ComponentSearchState state = pending_.top();
      pending_.pop();
      if (coversMask(state.covered, componentMask_))
        sealedCovers_.push_back(std::move(state.selectedRows));
    }
    llvm::sort(sealedCovers_, [&](const auto &lhs, const auto &rhs) {
      if (lhs.size() != rhs.size())
        return lhs.size() < rhs.size();
      const auto supplyBreadth = [&](const auto &rows) {
        std::uint64_t result = 0;
        for (const std::size_t row : rows)
          result = saturatingAdd(result, rowSupplyBreadth(domain_.rows[row]));
        return result;
      };
      const std::uint64_t lhsBreadth = supplyBreadth(lhs);
      const std::uint64_t rhsBreadth = supplyBreadth(rhs);
      if (lhsBreadth != rhsBreadth)
        return lhsBreadth > rhsBreadth;
      return std::lexicographical_compare(
          lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
          [&](std::size_t lhsRow, std::size_t rhsRow) {
            return domain_.rows[lhsRow].key < domain_.rows[rhsRow].key;
          });
    });
    sealedCovers_.erase(std::unique(sealedCovers_.begin(), sealedCovers_.end()),
                        sealedCovers_.end());
  }

  ComponentAdvance nextSealed() {
    if (executionControl_.stopRequested())
      return {ComponentAdvanceKind::Interrupted, {}};
    if (sealedCursor_ == sealedCovers_.size()) {
      exhausted_ = true;
      return {ComponentAdvanceKind::Exhausted, {}};
    }
    std::vector<const TechMatchRow *> cover;
    for (std::size_t row : sealedCovers_[sealedCursor_++])
      cover.push_back(&domain_.rows[row]);
    return {ComponentAdvanceKind::Cover, std::move(cover)};
  }

  const TechMatchDomain &domain_;
  const IncidenceComponent &component_;
  llvm::ArrayRef<std::vector<std::size_t>> rowsByActor_;
  llvm::ArrayRef<ActorMask> rowMasks_;
  llvm::ArrayRef<std::size_t> actorLocalSlots_;
  ActorMask componentMask_;
  const ResolvedTechMappingConfigView &config_;
  TechMappingGenerationAccounting &accounting_;
  ExecutionControlView executionControl_;
  std::priority_queue<ComponentSearchState, std::vector<ComponentSearchState>,
                      ComponentStateGreater>
      pending_;
  bool initialized_ = false;
  bool exhausted_ = false;
  bool sealed_ = false;
  bool truncated_ = false;
  std::vector<std::vector<std::size_t>> sealedCovers_;
  std::size_t sealedCursor_ = 0;
};

struct LazyComponentCovers final {
  std::unique_ptr<ComponentCoverCursor> cursor;
  std::vector<std::vector<const TechMatchRow *>> discovered;
  bool exhausted = false;
};

using SparseProductIndex = std::vector<std::pair<std::size_t, std::size_t>>;

std::size_t componentIndex(const SparseProductIndex &indices,
                           std::size_t component) {
  auto found = llvm::lower_bound(
      indices, component,
      [](const auto &entry, std::size_t value) { return entry.first < value; });
  return found != indices.end() && found->first == component ? found->second
                                                             : 0;
}

SparseProductIndex incrementComponent(const SparseProductIndex &indices,
                                      std::size_t component) {
  SparseProductIndex next = indices;
  auto found = llvm::lower_bound(
      next, component,
      [](const auto &entry, std::size_t value) { return entry.first < value; });
  if (found == next.end() || found->first != component)
    next.insert(found, {component, 1});
  else
    ++found->second;
  return next;
}

template <typename Callback>
void forEachDifferingComponent(const SparseProductIndex &lhs,
                               const SparseProductIndex &rhs,
                               Callback callback) {
  std::size_t lhsCursor = 0;
  std::size_t rhsCursor = 0;
  while (lhsCursor != lhs.size() || rhsCursor != rhs.size()) {
    const std::size_t lhsComponent =
        lhsCursor == lhs.size() ? std::numeric_limits<std::size_t>::max()
                                : lhs[lhsCursor].first;
    const std::size_t rhsComponent =
        rhsCursor == rhs.size() ? std::numeric_limits<std::size_t>::max()
                                : rhs[rhsCursor].first;
    const std::size_t component = std::min(lhsComponent, rhsComponent);
    std::size_t lhsIndex = 0;
    std::size_t rhsIndex = 0;
    if (lhsComponent == component)
      lhsIndex = lhs[lhsCursor++].second;
    if (rhsComponent == component)
      rhsIndex = rhs[rhsCursor++].second;
    if (lhsIndex != rhsIndex)
      callback(component, lhsIndex, rhsIndex);
  }
}

struct ProductState final {
  SparseProductIndex indices;
  std::size_t rowCount = 0;
  std::uint64_t supplyBreadth = 0;
};

struct ProductGreater final {
  const std::vector<LazyComponentCovers> *components;

  std::vector<const TechMatchRow *>
  differingRows(const ProductState &selected, const ProductState &other) const {
    std::vector<const TechMatchRow *> rows;
    forEachDifferingComponent(
        selected.indices, other.indices,
        [&](std::size_t component, std::size_t selectedIndex, std::size_t) {
          const auto &cover =
              (*components)[component].discovered[selectedIndex];
          rows.insert(rows.end(), cover.begin(), cover.end());
        });
    llvm::sort(rows, [](const TechMatchRow *lhs, const TechMatchRow *rhs) {
      return lhs->key < rhs->key;
    });
    return rows;
  }

  bool operator()(const ProductState &lhs, const ProductState &rhs) const {
    if (lhs.rowCount != rhs.rowCount)
      return lhs.rowCount > rhs.rowCount;
    if (lhs.supplyBreadth != rhs.supplyBreadth)
      return lhs.supplyBreadth < rhs.supplyBreadth;
    const std::vector<const TechMatchRow *> lhsRows = differingRows(lhs, rhs);
    const std::vector<const TechMatchRow *> rhsRows = differingRows(rhs, lhs);
    std::size_t lhsRow = 0;
    std::size_t rhsRow = 0;
    while (lhsRow != lhsRows.size() && rhsRow != rhsRows.size()) {
      if (lhsRows[lhsRow] == rhsRows[rhsRow]) {
        ++lhsRow;
        ++rhsRow;
        continue;
      }
      return lhsRows[lhsRow]->key > rhsRows[rhsRow]->key;
    }
    if (lhsRow != lhsRows.size() || rhsRow != rhsRows.size())
      return lhsRow != lhsRows.size();
    return lhs.indices > rhs.indices;
  }
};

std::vector<const TechMatchRow *>
materializeCover(const std::vector<LazyComponentCovers> &components,
                 const ProductState &state) {
  std::vector<const TechMatchRow *> cover;
  for (std::size_t component = 0; component < components.size(); ++component) {
    const auto &selected =
        components[component]
            .discovered[componentIndex(state.indices, component)];
    cover.insert(cover.end(), selected.begin(), selected.end());
  }
  llvm::sort(cover, [](const TechMatchRow *lhs, const TechMatchRow *rhs) {
    return lhs->key < rhs->key;
  });
  return cover;
}

} // namespace

TechCoverSearchResult
searchTechMatchCovers(const TechMatchDomain &domain,
                      const ResolvedTechMappingConfigView &config,
                      TechMappingGenerationAccounting &accounting,
                      ExecutionControlView executionControl) {
  return searchTechMatchCovers(domain, config, accounting,
                               config.candidatePublicationLimit(),
                               executionControl);
}

TechCoverSearchResult searchTechMatchCovers(
    const TechMatchDomain &domain, const ResolvedTechMappingConfigView &config,
    TechMappingGenerationAccounting &accounting, std::uint64_t coverLimit,
    ExecutionControlView executionControl) {
  TechCoverSearchResult result;
  if (executionControl.stopRequested()) {
    result.exhausted = false;
    result.interrupted = true;
    return result;
  }
  const std::vector<IncidenceComponent> incidence = componentsOf(domain);
  std::vector<std::vector<std::size_t>> rowsByActor(domain.actors.size());
  std::vector<ActorMask> rowMasks(domain.rows.size());
  std::vector<std::size_t> actorLocalSlots(domain.actors.size());
  for (auto [rowIndex, row] : llvm::enumerate(domain.rows))
    for (std::size_t actor : row.actorSlots)
      rowsByActor[actor].push_back(rowIndex);

  for (const IncidenceComponent &component : incidence) {
    const std::size_t words = actorMaskWordCount(component.actors.size());
    for (auto [local, actor] : llvm::enumerate(component.actors))
      actorLocalSlots[actor] = local;
    for (std::size_t row : component.rows) {
      rowMasks[row].assign(words, 0);
      for (std::size_t actor : domain.rows[row].actorSlots)
        setActor(rowMasks[row], actorLocalSlots[actor]);
    }
  }

  std::vector<LazyComponentCovers> components;
  components.reserve(incidence.size());
  for (const IncidenceComponent &component : incidence) {
    LazyComponentCovers lazy;
    lazy.cursor = std::make_unique<ComponentCoverCursor>(
        domain, component, rowsByActor, rowMasks, actorLocalSlots, config,
        accounting, executionControl);
    ComponentAdvance first = lazy.cursor->next();
    if (first.kind == ComponentAdvanceKind::Interrupted) {
      result.exhausted = false;
      result.interrupted = true;
      return result;
    }
    if (lazy.cursor->truncated())
      result.exhausted = false;
    if (first.kind == ComponentAdvanceKind::Exhausted) {
      lazy.exhausted = true;
    } else {
      lazy.discovered.push_back(std::move(first.cover));
    }
    components.push_back(std::move(lazy));
  }

  if (llvm::any_of(components, [](const auto &component) {
        return component.discovered.empty();
      })) {
    for (LazyComponentCovers &component : components) {
      while (!component.exhausted) {
        ComponentAdvance advance = component.cursor->next();
        if (advance.kind == ComponentAdvanceKind::Interrupted) {
          result.exhausted = false;
          result.interrupted = true;
          return result;
        }
        if (component.cursor->truncated())
          result.exhausted = false;
        component.exhausted = advance.kind == ComponentAdvanceKind::Exhausted;
        if (advance.kind == ComponentAdvanceKind::Cover)
          component.discovered.push_back(std::move(advance.cover));
      }
    }
    return result;
  }

  ProductState initial;
  for (const LazyComponentCovers &component : components) {
    initial.rowCount += component.discovered.front().size();
    initial.supplyBreadth =
        saturatingAdd(initial.supplyBreadth,
                      coverSupplyBreadth(component.discovered.front()));
  }
  std::priority_queue<ProductState, std::vector<ProductState>, ProductGreater>
      pending(ProductGreater{&components});
  pending.push(initial);
  std::set<SparseProductIndex> visited{{}};
  std::uint64_t fullSupplyChecks = 0;
  while (!pending.empty()) {
    if (executionControl.stopRequested()) {
      result.exhausted = false;
      result.interrupted = true;
      return result;
    }
    ProductState current = pending.top();
    pending.pop();
    std::vector<const TechMatchRow *> currentCover =
        materializeCover(components, current);
    if (fullSupplyChecks >= config.candidateEvaluationLimit()) {
      result.exhausted = false;
      return result;
    }
    ++fullSupplyChecks;
    const std::size_t computeCount =
        llvm::count_if(currentCover, [](const auto *row) {
          return std::holds_alternative<TechComputeRealizationView>(
              row->realization);
        });
    bool supplyAdmissible = true;
    if (computeCount > 1) {
      const SpatialComputeContextSupplyAnalysis supply =
          analyzeComputeContextSupply(currentCover, domain, accounting);
      supplyAdmissible = supply.admissible();
      if (!supplyAdmissible)
        emitComputeContextRejection(fullSupplyChecks - 1, currentCover, supply);
    }
    const std::size_t memoryCount =
        llvm::count_if(currentCover, [](const auto *row) {
          return std::holds_alternative<TechMemoryRealizationView>(
              row->realization);
        });
    if (supplyAdmissible && memoryCount > 1) {
      const SpatialMemoryOccurrenceSupplyAnalysis supply =
          analyzeMemoryOccurrenceSupply(currentCover, accounting,
                                        MemorySupplyCheckScope::FullCover);
      supplyAdmissible = supply.admissible();
      if (!supplyAdmissible)
        emitMemorySupplyRejection(fullSupplyChecks - 1, currentCover, supply);
    }
    if (supplyAdmissible)
      result.covers.push_back(std::move(currentCover));
    for (std::size_t component = 0; component < components.size();
         ++component) {
      const std::size_t currentIndex =
          componentIndex(current.indices, component);
      SparseProductIndex next = incrementComponent(current.indices, component);
      if (!visited.insert(next).second)
        continue;
      LazyComponentCovers &lazy = components[component];
      const std::size_t nextIndex = currentIndex + 1;
      if (nextIndex == lazy.discovered.size() && !lazy.exhausted) {
        ComponentAdvance advance = lazy.cursor->next();
        if (advance.kind == ComponentAdvanceKind::Interrupted) {
          result.exhausted = false;
          result.interrupted = true;
          return result;
        }
        if (lazy.cursor->truncated())
          result.exhausted = false;
        if (advance.kind == ComponentAdvanceKind::Exhausted)
          lazy.exhausted = true;
        else
          lazy.discovered.push_back(std::move(advance.cover));
      }
      if (nextIndex >= lazy.discovered.size())
        continue;
      const auto &currentCover = lazy.discovered[currentIndex];
      const auto &nextCover = lazy.discovered[nextIndex];
      const std::uint64_t currentComponentBreadth =
          coverSupplyBreadth(currentCover);
      const std::uint64_t nextComponentBreadth = coverSupplyBreadth(nextCover);
      const std::uint64_t baseBreadth =
          current.supplyBreadth == std::numeric_limits<std::uint64_t>::max()
              ? current.supplyBreadth
              : current.supplyBreadth - currentComponentBreadth;
      pending.push(ProductState{
          std::move(next),
          current.rowCount - currentCover.size() + nextCover.size(),
          saturatingAdd(baseBreadth, nextComponentBreadth)});
    }
    if (result.covers.size() >= coverLimit) {
      result.exhausted =
          result.exhausted && pending.empty() &&
          llvm::all_of(components, [](const LazyComponentCovers &component) {
            return component.exhausted;
          });
      return result;
    }
  }
  return result;
}

} // namespace loom::mapping::detail
