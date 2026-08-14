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
    return {vertices,        domainOffsets, domains,
            intervalOffsets, intervals,     domainCount};
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

void emptyRestrictionRemainsUnassigned() {
  OwnedProblem problem = makeProblem(1, 3, {{}}, 0);
  problem.vertices.front().restricted = true;
  const auto result =
      take(loom::pnr::detail::colorSpatialTagInterference(problem.view()));
  require(result.unassignedCount == 1 && result.conflictCount == 0 &&
              !result.values.front(),
          "empty allowed set was not retained as TagUnassigned");
}

} // namespace

int main() {
  exactColoringUsesLocalInterference();
  largeColoringMinimizesUnavoidableConflict();
  emptyRestrictionRemainsUnassigned();
  llvm::outs() << "spatial tag coloring tests passed\n";
  return EXIT_SUCCESS;
}
