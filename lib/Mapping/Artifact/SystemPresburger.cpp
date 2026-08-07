#include "Mapping/Artifact/SystemPresburger.h"

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/IR/AffineMap.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>

namespace loom::mapping {
namespace {

using mlir::presburger::IntegerPolyhedron;
using mlir::presburger::IntegerRelation;
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
      cell.dimensionCount, cell.symbolCount, cell.localCount);
  IntegerPolyhedron polyhedron(space);
  const std::size_t expectedWidth =
      static_cast<std::size_t>(cell.dimensionCount) + cell.symbolCount +
      cell.localCount + 1;
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

SystemPresburgerCell cellFromPolyhedron(const IntegerPolyhedron &polyhedron) {
  SystemPresburgerCell cell;
  cell.dimensionCount = polyhedron.getNumDimVars();
  cell.symbolCount = polyhedron.getNumSymbolVars();
  cell.localCount = polyhedron.getNumLocalVars();
  for (unsigned index = 0; index < polyhedron.getNumEqualities(); ++index) {
    const auto row = polyhedron.getEquality64(index);
    cell.equalities.emplace_back(row.begin(), row.end());
  }
  for (unsigned index = 0; index < polyhedron.getNumInequalities(); ++index) {
    const auto row = polyhedron.getInequality64(index);
    cell.inequalities.emplace_back(row.begin(), row.end());
  }
  return cell;
}

llvm::Expected<std::vector<SystemPresburgerCell>>
cellsFromSet(const PresburgerSet &set) {
  std::vector<SystemPresburgerCell> cells;
  cells.reserve(set.getNumDisjuncts());
  for (const IntegerRelation &disjunct : set.getAllDisjuncts()) {
    if (disjunct.isIntegerEmpty())
      continue;
    auto cell = canonicalizeSystemPresburgerCell(
        cellFromPolyhedron(IntegerPolyhedron(disjunct)));
    if (!cell)
      return cell.takeError();
    cells.push_back(std::move(*cell));
  }
  llvm::sort(cells, [](const SystemPresburgerCell &lhs,
                       const SystemPresburgerCell &rhs) {
    return std::tie(lhs.dimensionCount, lhs.symbolCount, lhs.localCount,
                    lhs.equalities, lhs.inequalities) <
           std::tie(rhs.dimensionCount, rhs.symbolCount, rhs.localCount,
                    rhs.equalities, rhs.inequalities);
  });
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
  return cells;
}

} // namespace

llvm::Expected<SystemPresburgerCell>
canonicalizeSystemPresburgerCell(const SystemPresburgerCell &input) {
  SystemPresburgerCell normalized = input;
  const std::size_t expectedWidth =
      static_cast<std::size_t>(normalized.dimensionCount) +
      normalized.symbolCount + normalized.localCount + 1;
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

  normalized.localCount = polyhedron->getNumLocalVars();
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

llvm::Expected<SystemPresburgerCell>
imageSystemPresburgerCell(const SystemPresburgerCell &input,
                          const mlir::AffineMap &map) {
  auto normalized = canonicalizeSystemPresburgerCell(input);
  if (!normalized)
    return normalized.takeError();
  if (map.getNumDims() != normalized->dimensionCount)
    return invalid("affine image has a foreign input dimension count");
  if (map.getNumSymbols() != 0)
    return invalid("affine image map must not contain symbols");

  mlir::AffineMap mutableMap = map;
  IntegerRelation relation(PresburgerSpace::getRelationSpace());
  if (mlir::failed(mlir::affine::getRelationFromMap(mutableMap, relation)))
    return invalid("affine image map is not Presburger-representable");
  relation.setSpaceExceptLocals(PresburgerSpace::getRelationSpace(
      map.getNumDims(), map.getNumResults(), /*numSymbols=*/0));

  const std::uint64_t inputLocalCount =
      static_cast<std::uint64_t>(normalized->symbolCount) +
      normalized->localCount;
  if (inputLocalCount > std::numeric_limits<unsigned>::max())
    return invalid("affine image input local count exceeds native range");
  relation.insertVar(mlir::presburger::VarKind::Local, /*pos=*/0,
                     static_cast<unsigned>(inputLocalCount));
  const unsigned localOffset =
      relation.getVarKindOffset(mlir::presburger::VarKind::Local);
  const std::size_t expectedInputWidth =
      static_cast<std::size_t>(normalized->dimensionCount) +
      normalized->symbolCount + normalized->localCount + 1;
  const auto appendRows = [&](const auto &rows, bool equality) -> llvm::Error {
    for (const std::vector<std::int64_t> &row : rows) {
      if (row.size() != expectedInputWidth)
        return invalid("affine image input row has the wrong width");
      std::vector<std::int64_t> lifted(relation.getNumVars() + 1, 0);
      std::copy_n(row.begin(), normalized->dimensionCount, lifted.begin());
      std::copy_n(row.begin() + normalized->dimensionCount, inputLocalCount,
                  lifted.begin() + localOffset);
      lifted.back() = row.back();
      if (equality)
        relation.addEquality(lifted);
      else
        relation.addInequality(lifted);
    }
    return llvm::Error::success();
  };
  if (llvm::Error error = appendRows(normalized->equalities, true))
    return std::move(error);
  if (llvm::Error error = appendRows(normalized->inequalities, false))
    return std::move(error);

  return canonicalizeSystemPresburgerCell(
      cellFromPolyhedron(relation.getRangeSet()));
}

llvm::Expected<std::optional<SystemPresburgerCell>>
intersectSystemPresburgerCells(const SystemPresburgerCell &lhs,
                               const SystemPresburgerCell &rhs) {
  auto left = canonicalizeSystemPresburgerCell(lhs);
  if (!left)
    return left.takeError();
  auto right = canonicalizeSystemPresburgerCell(rhs);
  if (!right)
    return right.takeError();
  if (left->dimensionCount != right->dimensionCount ||
      left->symbolCount != right->symbolCount)
    return invalid("cannot intersect cells from different spaces");
  const std::uint64_t localCount =
      static_cast<std::uint64_t>(left->localCount) + right->localCount;
  if (localCount > std::numeric_limits<std::uint32_t>::max())
    return invalid("Presburger intersection local count exceeds u32");

  SystemPresburgerCell combined;
  combined.dimensionCount = left->dimensionCount;
  combined.symbolCount = left->symbolCount;
  combined.localCount = static_cast<std::uint32_t>(localCount);
  const std::size_t prefix =
      static_cast<std::size_t>(combined.dimensionCount) + combined.symbolCount;
  const auto appendRows = [&](const auto &rows, std::uint32_t ownLocals,
                              std::uint32_t precedingLocals, auto &output) {
    for (const std::vector<std::int64_t> &row : rows) {
      std::vector<std::int64_t> lifted(prefix + combined.localCount + 1, 0);
      std::copy_n(row.begin(), prefix, lifted.begin());
      std::copy_n(row.begin() + prefix, ownLocals,
                  lifted.begin() + prefix + precedingLocals);
      lifted.back() = row.back();
      output.push_back(std::move(lifted));
    }
  };
  appendRows(left->equalities, left->localCount, 0, combined.equalities);
  appendRows(right->equalities, right->localCount, left->localCount,
             combined.equalities);
  appendRows(left->inequalities, left->localCount, 0, combined.inequalities);
  appendRows(right->inequalities, right->localCount, left->localCount,
             combined.inequalities);
  auto polyhedron = makePolyhedron(combined);
  if (!polyhedron)
    return polyhedron.takeError();
  if (polyhedron->isIntegerEmpty())
    return std::optional<SystemPresburgerCell>{};
  auto canonical = canonicalizeSystemPresburgerCell(combined);
  if (!canonical)
    return canonical.takeError();
  return std::optional<SystemPresburgerCell>(std::move(*canonical));
}

llvm::Expected<SystemPresburgerSetSplit>
splitSystemPresburgerSet(llvm::ArrayRef<SystemPresburgerCell> domain,
                         llvm::ArrayRef<SystemPresburgerCell> predicate) {
  if (domain.empty())
    return SystemPresburgerSetSplit{};
  const PresburgerSpace space = PresburgerSpace::getSetSpace(
      domain.front().dimensionCount, domain.front().symbolCount,
      /*numLocals=*/0);
  auto domainSet = makeUnion(domain, space);
  if (!domainSet)
    return domainSet.takeError();
  auto predicateSet = makeUnion(predicate, space);
  if (!predicateSet)
    return predicateSet.takeError();
  auto inside = cellsFromSet(domainSet->intersect(*predicateSet).coalesce());
  if (!inside)
    return inside.takeError();
  auto outside = cellsFromSet(domainSet->subtract(*predicateSet).coalesce());
  if (!outside)
    return outside.takeError();
  return SystemPresburgerSetSplit{std::move(*inside), std::move(*outside)};
}

llvm::Expected<bool>
systemPresburgerCellsIntersect(const SystemPresburgerCell &lhs,
                               const SystemPresburgerCell &rhs) {
  auto intersection = intersectSystemPresburgerCells(lhs, rhs);
  if (!intersection)
    return intersection.takeError();
  return intersection->has_value();
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
