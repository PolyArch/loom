#include "Mapping/Artifact/SystemPresburger.h"

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>

namespace loom::mapping {
namespace {

using mlir::presburger::IntegerPolyhedron;
using mlir::presburger::PresburgerSet;
using mlir::presburger::PresburgerSpace;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_presburger_invalid: " + message);
}

std::uint64_t magnitude(std::int64_t value) {
  if (value >= 0)
    return static_cast<std::uint64_t>(value);
  return static_cast<std::uint64_t>(-(value + 1)) + 1;
}

llvm::Error normalizeRow(std::vector<std::int64_t> &row, bool equality) {
  std::uint64_t divisor = 0;
  for (std::int64_t value : row)
    divisor = std::gcd(divisor, magnitude(value));
  if (divisor == 0)
    return invalid(equality ? "zero equality row" : "zero inequality row");
  if (divisor != 1)
    for (std::int64_t &value : row)
      value = static_cast<std::int64_t>(static_cast<__int128>(value) /
                                        static_cast<__int128>(divisor));
  if (!equality)
    return llvm::Error::success();

  const auto first =
      llvm::find_if(row, [](std::int64_t value) { return value != 0; });
  if (first == row.end() || *first > 0)
    return llvm::Error::success();
  for (std::int64_t &value : row) {
    if (value == std::numeric_limits<std::int64_t>::min())
      return invalid("equality sign normalization overflows i64");
    value = -value;
  }
  return llvm::Error::success();
}

llvm::Expected<IntegerPolyhedron>
makePolyhedron(const SystemPresburgerCell &cell) {
  const PresburgerSpace space = PresburgerSpace::getSetSpace(
      cell.dimensionCount, cell.symbolCount, /*numLocals=*/0);
  IntegerPolyhedron polyhedron(space);
  const std::size_t expectedWidth =
      static_cast<std::size_t>(cell.dimensionCount) + cell.symbolCount + 1;
  for (const std::vector<std::int64_t> &row : cell.equalities) {
    if (row.size() != expectedWidth)
      return invalid("equality row has the wrong width");
    polyhedron.addEquality(row);
  }
  for (const std::vector<std::int64_t> &row : cell.inequalities) {
    if (row.size() != expectedWidth)
      return invalid("inequality row has the wrong width");
    polyhedron.addInequality(row);
  }
  return polyhedron;
}

llvm::Expected<PresburgerSet> makeSet(const SystemPresburgerCell &cell) {
  auto polyhedron = makePolyhedron(cell);
  if (!polyhedron)
    return polyhedron.takeError();
  return PresburgerSet(*polyhedron);
}

llvm::Expected<PresburgerSet>
makeUnion(llvm::ArrayRef<SystemPresburgerCell> cells,
          const PresburgerSpace &space) {
  PresburgerSet result = PresburgerSet::getEmpty(space);
  for (const SystemPresburgerCell &cell : cells) {
    auto candidate = makeSet(cell);
    if (!candidate)
      return candidate.takeError();
    if (!candidate->getSpace().isEqual(space))
      return invalid("cell has a foreign logical signature");
    result.unionInPlace(*candidate);
  }
  return result;
}

} // namespace

llvm::Expected<SystemPresburgerCell>
canonicalizeSystemPresburgerCell(const SystemPresburgerCell &input) {
  SystemPresburgerCell normalized = input;
  const std::size_t expectedWidth =
      static_cast<std::size_t>(normalized.dimensionCount) +
      normalized.symbolCount + 1;
  const auto removeTrivialRows = [&](auto &rows,
                                     const llvm::Twine &kind) -> llvm::Error {
    for (const std::vector<std::int64_t> &row : rows)
      if (row.size() != expectedWidth)
        return invalid(kind + " row has the wrong width");
    rows.erase(llvm::remove_if(rows,
                               [](const auto &row) {
                                 return llvm::all_of(row,
                                                     [](std::int64_t value) {
                                                       return value == 0;
                                                     });
                               }),
               rows.end());
    return llvm::Error::success();
  };
  if (llvm::Error error = removeTrivialRows(normalized.equalities, "equality"))
    return std::move(error);
  if (llvm::Error error =
          removeTrivialRows(normalized.inequalities, "inequality"))
    return std::move(error);
  for (std::vector<std::int64_t> &row : normalized.equalities)
    if (llvm::Error error = normalizeRow(row, /*equality=*/true))
      return std::move(error);
  for (std::vector<std::int64_t> &row : normalized.inequalities)
    if (llvm::Error error = normalizeRow(row, /*equality=*/false))
      return std::move(error);
  llvm::sort(normalized.equalities);
  llvm::sort(normalized.inequalities);
  normalized.equalities.erase(
      std::unique(normalized.equalities.begin(), normalized.equalities.end()),
      normalized.equalities.end());
  normalized.inequalities.erase(std::unique(normalized.inequalities.begin(),
                                            normalized.inequalities.end()),
                                normalized.inequalities.end());

  auto polyhedron = makePolyhedron(normalized);
  if (!polyhedron)
    return polyhedron.takeError();
  polyhedron->removeTrivialRedundancy();
  polyhedron->removeRedundantConstraints();
  if (polyhedron->isIntegerEmpty())
    return invalid("Presburger cell is integer-empty");

  normalized.equalities.clear();
  normalized.inequalities.clear();
  for (unsigned index = 0; index < polyhedron->getNumEqualities(); ++index) {
    auto row = polyhedron->getEquality64(index);
    normalized.equalities.emplace_back(row.begin(), row.end());
  }
  for (unsigned index = 0; index < polyhedron->getNumInequalities(); ++index) {
    auto row = polyhedron->getInequality64(index);
    normalized.inequalities.emplace_back(row.begin(), row.end());
  }
  for (std::vector<std::int64_t> &row : normalized.equalities)
    if (llvm::Error error = normalizeRow(row, /*equality=*/true))
      return std::move(error);
  for (std::vector<std::int64_t> &row : normalized.inequalities)
    if (llvm::Error error = normalizeRow(row, /*equality=*/false))
      return std::move(error);
  llvm::sort(normalized.equalities);
  llvm::sort(normalized.inequalities);
  return normalized;
}

llvm::Expected<bool>
systemPresburgerCellsIntersect(const SystemPresburgerCell &lhs,
                               const SystemPresburgerCell &rhs) {
  if (lhs.dimensionCount != rhs.dimensionCount ||
      lhs.symbolCount != rhs.symbolCount)
    return invalid("cannot intersect cells from different spaces");
  auto left = makeSet(lhs);
  if (!left)
    return left.takeError();
  auto right = makeSet(rhs);
  if (!right)
    return right.takeError();
  return !left->intersect(*right).isIntegerEmpty();
}

llvm::Expected<SystemPresburgerPartitionAnalysis>
analyzeSystemPresburgerPartition(llvm::ArrayRef<SystemPresburgerCell> cells,
                                 const SystemPresburgerCell &legalDomain) {
  auto legal = makeSet(legalDomain);
  if (!legal)
    return legal.takeError();
  auto covered = makeUnion(cells, legal->getSpace());
  if (!covered)
    return covered.takeError();

  bool disjoint = true;
  PresburgerSet accumulated = PresburgerSet::getEmpty(legal->getSpace());
  for (const SystemPresburgerCell &cell : cells) {
    auto candidate = makeSet(cell);
    if (!candidate)
      return candidate.takeError();
    if (!accumulated.intersect(*candidate).isIntegerEmpty())
      disjoint = false;
    accumulated.unionInPlace(*candidate);
  }
  return SystemPresburgerPartitionAnalysis{covered->isSubsetOf(*legal),
                                           disjoint, covered->isEqual(*legal)};
}

llvm::Expected<bool>
systemPresburgerSetIsSubsetOf(llvm::ArrayRef<SystemPresburgerCell> subset,
                              llvm::ArrayRef<SystemPresburgerCell> superset) {
  if (subset.empty())
    return true;
  const PresburgerSpace space = PresburgerSpace::getSetSpace(
      subset.front().dimensionCount, subset.front().symbolCount,
      /*numLocals=*/0);
  auto left = makeUnion(subset, space);
  if (!left)
    return left.takeError();
  auto right = makeUnion(superset, space);
  if (!right)
    return right.takeError();
  return left->isSubsetOf(*right);
}

} // namespace loom::mapping
