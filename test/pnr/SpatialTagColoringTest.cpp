#include "SpatialTagColoring.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "spatial tag coloring test failed: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

struct OwnedProblem final {
  std::vector<loom::pnr::detail::SpatialTagColoringVertex> vertices;
  std::vector<loom::pnr::PnrIndex> domainOffsets;
  std::vector<loom::pnr::PnrIndex> domains;
  std::vector<loom::pnr::PnrIndex> intervalOffsets;
  std::vector<loom::pnr::detail::SpatialTagColoringInterval> intervals;
  loom::pnr::PnrIndex domainCount = 0;

  loom::pnr::detail::SpatialTagColoringProblemView view() const {
    return {vertices,  domainOffsets, domains, intervalOffsets,
            intervals, domainCount,   {},      {}};
  }
};

OwnedProblem makeProblem(std::size_t vertexCount, std::uint32_t tagWidthBits,
                         std::vector<std::vector<loom::pnr::PnrIndex>> domains,
                         loom::pnr::PnrIndex domainCount) {
  OwnedProblem result;
  result.vertices.assign(vertexCount, {tagWidthBits, false});
  result.domainOffsets.push_back(0);
  result.intervalOffsets.assign(vertexCount + 1, 0);
  result.domainCount = domainCount;
  for (auto &local : domains) {
    llvm::sort(local);
    local.erase(std::unique(local.begin(), local.end()), local.end());
    result.domains.insert(result.domains.end(), local.begin(), local.end());
    result.domainOffsets.push_back(result.domains.size());
  }
  return result;
}

std::uint64_t value(const std::optional<llvm::APInt> &candidate) {
  require(candidate.has_value(), "expected an assigned Physical Tag");
  return candidate->getZExtValue();
}

std::vector<loom::pnr::detail::SpatialTagColoringVertexIdentity>
makeIdentities(std::initializer_list<std::uint64_t> owners) {
  std::vector<loom::pnr::detail::SpatialTagColoringVertexIdentity> result;
  std::optional<std::uint64_t> previousOwner;
  std::uint64_t origin = 0;
  for (std::uint64_t owner : owners) {
    if (!previousOwner || *previousOwner != owner)
      origin = 0;
    result.push_back({owner, 0, origin++});
    previousOwner = owner;
  }
  return result;
}

void exactColoringUsesLocalInterference() {
  std::vector<std::vector<loom::pnr::PnrIndex>> domains(6);
  loom::pnr::PnrIndex domain = 0;
  for (loom::pnr::PnrIndex lhs = 0; lhs < 3; ++lhs)
    for (loom::pnr::PnrIndex rhs = 3; rhs < 6; ++rhs) {
      if (rhs - 3 == lhs)
        continue;
      domains[lhs].push_back(domain);
      domains[rhs].push_back(domain);
      ++domain;
    }
  const OwnedProblem problem = makeProblem(6, 2, std::move(domains), domain);
  const auto result =
      take(loom::pnr::detail::colorSpatialTagInterference(problem.view()));
  require(result.unassignedCount == 0 && result.conflictCount == 0,
          "small exact coloring did not close a two-colorable component");
  std::set<std::uint64_t> colors;
  for (const auto &candidate : result.values)
    colors.insert(value(candidate));
  require(colors.size() == 2 && *colors.begin() == 0,
          "small exact coloring did not use a low contiguous palette");

  std::vector<std::vector<loom::pnr::PnrIndex>> localDomains(8);
  for (loom::pnr::PnrIndex vertex = 0; vertex < 4; ++vertex)
    localDomains[vertex].push_back(0);
  for (loom::pnr::PnrIndex vertex = 4; vertex < 8; ++vertex)
    localDomains[vertex].push_back(1);
  const OwnedProblem local = makeProblem(8, 2, std::move(localDomains), 2);
  const auto localResult =
      take(loom::pnr::detail::colorSpatialTagInterference(local.view()));
  require(localResult.conflictCount == 0,
          "disjoint local tag domains acquired a collision");
  for (std::size_t ordinal = 0; ordinal < 4; ++ordinal)
    require(value(localResult.values[ordinal]) ==
                value(localResult.values[ordinal + 4]),
            "disjoint local domains did not reuse their low palette");
}

void largeColoringMinimizesUnavoidableConflict() {
  const std::size_t vertexCount =
      loom::pnr::detail::spatialTagExactColoringVertexLimit + 1;
  std::vector<std::vector<loom::pnr::PnrIndex>> domains(vertexCount, {0});
  const OwnedProblem problem =
      makeProblem(vertexCount, 6, std::move(domains), 1);
  const auto result =
      take(loom::pnr::detail::colorSpatialTagInterference(problem.view()));
  require(result.unassignedCount == 0 && result.conflictCount == 1,
          "large heuristic coloring did not attain the clique lower bound");
  std::set<std::uint64_t> colors;
  for (const auto &candidate : result.values)
    colors.insert(value(candidate));
  require(colors.size() == 64 && *colors.begin() == 0 && *colors.rbegin() == 63,
          "large heuristic coloring did not use the representable palette");
}

void explicitInterferencePermitsPackedDomainReuse() {
  OwnedProblem problem = makeProblem(3, 2, {{0}, {0}, {0}}, 1);
  const std::vector<loom::pnr::PnrIndex> conflictOffsets{0, 1, 2, 4};
  const std::vector<loom::pnr::PnrIndex> conflicts{2, 2, 0, 1};
  auto view = problem.view();
  view.vertexConflictOffsets = conflictOffsets;
  view.vertexConflicts = conflicts;
  const auto result =
      take(loom::pnr::detail::colorSpatialTagInterference(view));
  require(result.unassignedCount == 0 && result.conflictCount == 0,
          "explicit interference introduced a false tag conflict");
  require(value(result.values[0]) == value(result.values[1]) &&
              value(result.values[0]) != value(result.values[2]),
          "compatible domain members did not share one Physical Tag");
}

void emptyRestrictionRemainsUnassigned() {
  OwnedProblem problem = makeProblem(1, 3, {{}}, 0);
  problem.vertices.front().restricted = true;
  const auto result =
      take(loom::pnr::detail::colorSpatialTagInterference(problem.view()));
  require(result.unassignedCount == 1 && result.conflictCount == 0 &&
              !result.values.front(),
          "empty allowed set was not retained as TagUnassigned");
}

void cachedColoringSurvivesFlattenedOrdinalChanges() {
  const OwnedProblem empty = makeProblem(0, 2, {}, 0);
  const auto emptyInitial =
      take(loom::pnr::detail::colorSpatialTagInterference(empty.view(), {}));
  const auto emptyCached = take(loom::pnr::detail::colorSpatialTagInterference(
      empty.view(), {}, &emptyInitial.cache));
  require(emptyCached.values.empty() && emptyCached.cache.components.empty(),
          "empty coloring rejected an empty incremental cache");

  OwnedProblem initial = makeProblem(4, 2, {{0}, {0}, {1}, {1}}, 2);
  initial.vertices[0].restricted = true;
  initial.vertices[1].restricted = true;
  const std::vector<loom::pnr::PnrIndex> initialConflictOffsets{0, 1, 2, 3, 4};
  const std::vector<loom::pnr::PnrIndex> initialConflicts{1, 0, 3, 2};
  auto initialView = initial.view();
  initialView.vertexConflictOffsets = initialConflictOffsets;
  initialView.vertexConflicts = initialConflicts;
  const auto initialIdentities = makeIdentities({0, 0, 1, 1});
  const auto initialResult =
      take(loom::pnr::detail::colorSpatialTagInterference(initialView,
                                                          initialIdentities));
  require(initialResult.cache.components.size() == 2,
          "initial coloring did not retain two exact components");

  OwnedProblem updated = makeProblem(5, 2, {{0}, {0}, {0}, {1}, {1}}, 2);
  updated.vertices[0].restricted = true;
  updated.vertices[1].restricted = true;
  updated.vertices[2].restricted = true;
  const std::vector<loom::pnr::PnrIndex> updatedConflictOffsets{0, 2, 4,
                                                                6, 7, 8};
  const std::vector<loom::pnr::PnrIndex> updatedConflicts{1, 2, 0, 2,
                                                          0, 1, 4, 3};
  auto updatedView = updated.view();
  updatedView.vertexConflictOffsets = updatedConflictOffsets;
  updatedView.vertexConflicts = updatedConflicts;
  const auto updatedIdentities = makeIdentities({0, 0, 0, 1, 1});
  const auto cached = take(loom::pnr::detail::colorSpatialTagInterference(
      updatedView, updatedIdentities, &initialResult.cache));
  const auto cold = take(loom::pnr::detail::colorSpatialTagInterference(
      updatedView, updatedIdentities));
  const bool equalValues = llvm::equal(
      cached.values, cold.values, [](const auto &lhs, const auto &rhs) {
        return lhs.has_value() == rhs.has_value() && (!lhs || *lhs == *rhs);
      });
  require(equalValues && cached.unassignedCount == cold.unassignedCount &&
              cached.conflictCount == cold.conflictCount,
          "cached coloring diverged after flattened ordinals changed");
  require(cached.cache.components.size() == cold.cache.components.size(),
          "cached coloring lost its canonical component inventory");
  require(
      cached.recomputedIdentities.size() == 3 &&
          llvm::all_of(cached.recomputedIdentities,
                       [](const auto identity) { return identity.owner == 0; }),
      "unchanged component was recolored after its ordinal shifted");

  const OwnedProblem workInitial = makeProblem(4, 2, {{0}, {0}, {1}, {1}}, 2);
  auto workInitialView = workInitial.view();
  workInitialView.vertexConflictOffsets = initialConflictOffsets;
  workInitialView.vertexConflicts = initialConflicts;
  const auto workInitialResult =
      take(loom::pnr::detail::colorSpatialTagInterference(workInitialView,
                                                          initialIdentities));
  const OwnedProblem workUpdated =
      makeProblem(5, 2, {{0}, {0}, {0}, {1}, {1}}, 2);
  auto workUpdatedView = workUpdated.view();
  workUpdatedView.vertexConflictOffsets = updatedConflictOffsets;
  workUpdatedView.vertexConflicts = updatedConflicts;
  const auto workCached = take(loom::pnr::detail::colorSpatialTagInterference(
      workUpdatedView, updatedIdentities, &workInitialResult.cache));
  const auto workCold = take(loom::pnr::detail::colorSpatialTagInterference(
      workUpdatedView, updatedIdentities));
  const bool equalWorkValues = llvm::equal(
      workCached.values, workCold.values, [](const auto &lhs, const auto &rhs) {
        return lhs.has_value() == rhs.has_value() && (!lhs || *lhs == *rhs);
      });
  require(workInitialResult.cache.components[1].exactWorkBefore !=
                  workCold.cache.components[1].exactWorkBefore &&
              workCached.recomputedIdentities.size() ==
                  updatedIdentities.size() &&
              equalWorkValues &&
              workCached.unassignedCount == workCold.unassignedCount &&
              workCached.conflictCount == workCold.conflictCount,
          "coloring cache ignored changed exact-work prefix state");
}

} // namespace

int main() {
  exactColoringUsesLocalInterference();
  largeColoringMinimizesUnavoidableConflict();
  explicitInterferencePermitsPackedDomainReuse();
  emptyRestrictionRemainsUnassigned();
  cachedColoringSurvivesFlattenedOrdinalChanges();
  llvm::outs() << "spatial tag coloring tests passed\n";
  return EXIT_SUCCESS;
}
