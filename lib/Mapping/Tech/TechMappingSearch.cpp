#include "TechMappingCandidate.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <utility>
#include <vector>

namespace loom::mapping::detail {
namespace {

using ActorMask = std::vector<std::uint64_t>;

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
};

// A cover's row count is the number of physical realizations SpatialMapping
// must place: one FU capability instance per compute row and one Memory
// Operation Engine per memory row. Ordering partial states by canonical row
// keys alone selects the cover whose rows have the smallest keys, which is
// systematically the cover that binds one actor per realization and therefore
// demands the most occurrences. `lowerBound` already holds the selected rows
// unioned with one compatible row for every uncovered actor, so its size is a
// realization-demand estimate that the search can minimize first. This uses no
// occurrence count, coordinate, or other physical inventory fact.
struct ComponentStateGreater final {
  bool operator()(const ComponentSearchState &lhs,
                  const ComponentSearchState &rhs) const {
    if (lhs.lowerBound.size() != rhs.lowerBound.size())
      return lhs.lowerBound.size() > rhs.lowerBound.size();
    if (lhs.lowerBound != rhs.lowerBound)
      return lhs.lowerBound > rhs.lowerBound;
    return lhs.selectedRows > rhs.selectedRows;
  }
};

enum class ComponentAdvanceKind : std::uint8_t {
  Cover,
  Exhausted,
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
                       TechMappingGenerationAccounting &accounting)
      : domain_(domain), component_(component), rowsByActor_(rowsByActor),
        rowMasks_(rowMasks), actorLocalSlots_(actorLocalSlots),
        componentMask_(actorMaskWordCount(component.actors.size()), 0),
        config_(config), accounting_(accounting) {
    for (std::size_t actor = 0; actor < component_.actors.size(); ++actor)
      setActor(componentMask_, actor);
  }

  ComponentAdvance next() {
    if (exhausted_)
      return {ComponentAdvanceKind::Exhausted, {}};
    if (!initialized_)
      initialize();
    if (exhausted_)
      return {ComponentAdvanceKind::Exhausted, {}};
    if (sealed_)
      return nextSealed();

    while (!pending_.empty()) {
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
        if (!consumeExpansion()) {
          seal();
          return nextSealed();
        }
        ComponentSearchState child = state;
        addRow(child, row);
        const PropagationResult propagation = propagate(child);
        if (propagation == PropagationResult::LimitReached) {
          seal();
          return nextSealed();
        }
        if (propagation == PropagationResult::Viable && setLowerBound(child))
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
    for (std::size_t actor : component_.actors) {
      if (covered(state, actor))
        continue;
      const std::vector<std::size_t> compatible =
          compatibleRows(state, actor, 1);
      if (compatible.empty())
        return false;
      state.lowerBound.push_back(compatible.front());
    }
    llvm::sort(state.lowerBound);
    state.lowerBound.erase(
        std::unique(state.lowerBound.begin(), state.lowerBound.end()),
        state.lowerBound.end());
    return true;
  }

  void initialize() {
    initialized_ = true;
    ComponentSearchState initial{
        ActorMask(actorMaskWordCount(component_.actors.size()), 0), {}, {}};
    const PropagationResult propagation = propagate(initial);
    if (propagation == PropagationResult::LimitReached) {
      seal();
      return;
    }
    if (propagation == PropagationResult::Infeasible ||
        !setLowerBound(initial)) {
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
    llvm::sort(sealedCovers_);
    sealedCovers_.erase(std::unique(sealedCovers_.begin(), sealedCovers_.end()),
                        sealedCovers_.end());
  }

  ComponentAdvance nextSealed() {
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
};

struct ProductGreater final {
  const std::vector<LazyComponentCovers> *components;

  std::size_t rowCount(const ProductState &state) const {
    std::size_t count = 0;
    for (std::size_t component = 0; component < components->size();
         ++component)
      count += (*components)[component]
                   .discovered[componentIndex(state.indices, component)]
                   .size();
    return count;
  }

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
    const std::size_t lhsCount = rowCount(lhs);
    const std::size_t rhsCount = rowCount(rhs);
    if (lhsCount != rhsCount)
      return lhsCount > rhsCount;
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
                      TechMappingGenerationAccounting &accounting) {
  TechCoverSearchResult result;
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
        accounting);
    ComponentAdvance first = lazy.cursor->next();
    if (lazy.cursor->truncated())
      result.exhausted = false;
    if (first.kind == ComponentAdvanceKind::Exhausted)
      lazy.exhausted = true;
    else
      lazy.discovered.push_back(std::move(first.cover));
    components.push_back(std::move(lazy));
  }

  if (llvm::any_of(components, [](const auto &component) {
        return component.discovered.empty();
      })) {
    for (LazyComponentCovers &component : components) {
      while (!component.exhausted) {
        ComponentAdvance advance = component.cursor->next();
        if (component.cursor->truncated())
          result.exhausted = false;
        component.exhausted = advance.kind == ComponentAdvanceKind::Exhausted;
        if (advance.kind == ComponentAdvanceKind::Cover)
          component.discovered.push_back(std::move(advance.cover));
      }
    }
    return result;
  }

  std::priority_queue<ProductState, std::vector<ProductState>, ProductGreater>
      pending(ProductGreater{&components});
  pending.push(ProductState{{}});
  std::set<SparseProductIndex> visited{{}};
  while (!pending.empty()) {
    ProductState current = pending.top();
    pending.pop();
    result.covers.push_back(materializeCover(components, current));
    if (result.covers.size() >= config.candidatePublicationLimit()) {
      result.exhausted = false;
      return result;
    }

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
        if (lazy.cursor->truncated())
          result.exhausted = false;
        if (advance.kind == ComponentAdvanceKind::Exhausted)
          lazy.exhausted = true;
        else
          lazy.discovered.push_back(std::move(advance.cover));
      }
      if (nextIndex >= lazy.discovered.size())
        continue;
      pending.push(ProductState{std::move(next)});
    }
  }
  return result;
}

} // namespace loom::mapping::detail
