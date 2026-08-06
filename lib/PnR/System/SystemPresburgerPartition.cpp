#include "SystemPnrSearchDomainInternal.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace loom::pnr {
namespace {

using mlir::presburger::IntegerPolyhedron;
using mlir::presburger::PresburgerSet;
using mlir::presburger::PresburgerSpace;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_pnr_search_domain_invalid: " +
                                     message);
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
  if (divisor != 1) {
    for (std::int64_t &value : row) {
      const __int128 quotient =
          static_cast<__int128>(value) / static_cast<__int128>(divisor);
      value = static_cast<std::int64_t>(quotient);
    }
  }
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
      return invalid("Presburger equality row has the wrong width");
    polyhedron.addEquality(row);
  }
  for (const std::vector<std::int64_t> &row : cell.inequalities) {
    if (row.size() != expectedWidth)
      return invalid("Presburger inequality row has the wrong width");
    polyhedron.addInequality(row);
  }
  return polyhedron;
}

llvm::Expected<SystemPresburgerCell>
canonicalizeCell(const SystemPresburgerCell &input) {
  if (input.dimensionCount > std::numeric_limits<unsigned>::max() ||
      input.symbolCount > std::numeric_limits<unsigned>::max())
    return invalid("Presburger arity exceeds native verifier range");

  SystemPresburgerCell normalized = input;
  const std::size_t expectedWidth =
      static_cast<std::size_t>(normalized.dimensionCount) +
      normalized.symbolCount + 1;
  const auto removeTrivialRows = [&](auto &rows,
                                     const llvm::Twine &kind) -> llvm::Error {
    for (const std::vector<std::int64_t> &row : rows)
      if (row.size() != expectedWidth)
        return invalid("Presburger " + kind + " row has the wrong width");
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
    return invalid("Presburger partition contains an empty cell");

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

llvm::Expected<PresburgerSet> makeSet(const SystemPresburgerCell &cell) {
  auto polyhedron = makePolyhedron(cell);
  if (!polyhedron)
    return polyhedron.takeError();
  return PresburgerSet(*polyhedron);
}

llvm::Expected<SystemPresburgerCell>
denseMayDomain(const ::dataflow::CanonicalRootThreadLogicalDomainView &domain) {
  if (domain.kind == ::dataflow::ThreadDomainKind::DynamicWork)
    return llvm::make_error<UnsupportedSystemPnrSearchDomain>(
        UnsupportedSystemPnrSearchDomainReason::
            DynamicWorkStableKeyProjectionUnavailable,
        "Dataflow has no exact DynamicWork stable-key projection");
  if (domain.launchParameters.size() >
      std::numeric_limits<std::uint32_t>::max())
    return invalid("logical launch-parameter count exceeds u32");

  SystemPresburgerCell cell;
  cell.dimensionCount = domain.coordinateRank;
  cell.symbolCount = static_cast<std::uint32_t>(domain.launchParameters.size());
  const std::size_t rowWidth =
      static_cast<std::size_t>(cell.dimensionCount) + cell.symbolCount + 1;
  for (std::uint32_t coordinate = 0; coordinate < cell.dimensionCount;
       ++coordinate) {
    std::vector<std::int64_t> lower(rowWidth, 0);
    lower[coordinate] = 1;
    cell.inequalities.push_back(std::move(lower));

    std::vector<std::int64_t> upper(rowWidth, 0);
    upper[coordinate] = -1;
    upper[cell.dimensionCount + coordinate] = 1;
    upper.back() = -1;
    cell.inequalities.push_back(std::move(upper));
  }
  return canonicalizeCell(cell);
}

struct ExpectedBinding final {
  SystemSearchBindingKey key;
  SystemPresburgerCell legalDomain;
  std::vector<std::uint8_t> canonicalKey;
};

llvm::Expected<std::vector<ExpectedBinding>> collectExpectedBindings(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots) {
  std::vector<ExpectedBinding> expected;
  std::map<std::uint64_t, SystemPresburgerCell> legalDomainsByRoot;
  for (::dataflow::RootThreadLaunchRef root : roots) {
    auto logical = dataflow.projectRootThreadLogicalDomain(root);
    if (!logical)
      return logical.takeError();
    auto legalDomain = denseMayDomain(*logical);
    if (!legalDomain)
      return legalDomain.takeError();
    auto threadKey =
        detail::canonicalBindingKeyBytes(root, dataflow.identity());
    if (!threadKey)
      return threadKey.takeError();
    expected.push_back(
        {SystemSearchBindingKey(root), *legalDomain, std::move(*threadKey)});
    legalDomainsByRoot.emplace(root.entity.value(), std::move(*legalDomain));
  }

  llvm::Error callbackError = llvm::Error::success();
  dataflow.forEachRootedGraphLaunch(
      [&](::dataflow::RootedGraphLaunchRef graph) {
        if (callbackError)
          return;
        const auto domain =
            legalDomainsByRoot.find(graph.rootThreadLaunch.entity.value());
        if (domain == legalDomainsByRoot.end())
          return;
        auto graphLogical =
            dataflow.projectWholeRootedGraphLogicalDomain(graph);
        if (!graphLogical) {
          callbackError = graphLogical.takeError();
          return;
        }
        if (!*graphLogical) {
          callbackError = llvm::make_error<UnsupportedSystemPnrSearchDomain>(
              UnsupportedSystemPnrSearchDomainReason::
                  RootedGraphMayDomainProjectionUnavailable,
              "Canonical Dataflow does not publish the exact may-domain of a "
              "nested or repeated rooted graph launch");
          return;
        }
        auto graphKey =
            detail::canonicalBindingKeyBytes(graph, dataflow.identity());
        if (!graphKey) {
          callbackError = graphKey.takeError();
          return;
        }
        expected.push_back({SystemSearchBindingKey(graph), domain->second,
                            std::move(*graphKey)});
      });
  if (callbackError)
    return std::move(callbackError);

  llvm::sort(expected,
             [](const ExpectedBinding &lhs, const ExpectedBinding &rhs) {
               return lhs.canonicalKey < rhs.canonicalKey;
             });
  return expected;
}

llvm::Error validatePartition(llvm::ArrayRef<SystemPresburgerCell> cells,
                              const SystemPresburgerCell &legalDomain) {
  auto legal = makeSet(legalDomain);
  if (!legal)
    return legal.takeError();
  PresburgerSet covered = PresburgerSet::getEmpty(legal->getSpace());
  for (const SystemPresburgerCell &cell : cells) {
    auto candidate = makeSet(cell);
    if (!candidate)
      return candidate.takeError();
    if (!candidate->isSubsetOf(*legal))
      return invalid("Presburger cell extends beyond the Dataflow may-domain");
    if (!covered.intersect(*candidate).isIntegerEmpty())
      return invalid("Presburger partition cells overlap");
    covered.unionInPlace(*candidate);
  }
  if (!covered.isEqual(*legal))
    return invalid(
        "Presburger partition does not cover the Dataflow may-domain");
  return llvm::Error::success();
}

} // namespace

namespace detail {

llvm::Expected<std::vector<std::uint8_t>>
canonicalBindingKeyBytes(const SystemSearchBindingKey &key,
                         const ArtifactIdentity &dataflowIdentity) {
  std::vector<std::uint8_t> bytes;
  const std::uint32_t kind =
      std::holds_alternative<::dataflow::RootThreadLaunchRef>(key) ? 0 : 1;
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(kind >> shift));
  auto local = std::visit(
      [&](const auto &reference) {
        return ::dataflow::encodeDataflowReference(dataflowIdentity, reference);
      },
      key);
  if (!local)
    return local.takeError();
  bytes.insert(bytes.end(), local->begin(), local->end());
  return bytes;
}

llvm::Expected<std::vector<::dataflow::RootThreadLaunchRef>>
canonicalRootThreadLaunchSet(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots) {
  if (roots.empty())
    return invalid("root launch set is empty");
  std::vector<
      std::pair<std::vector<std::uint8_t>, ::dataflow::RootThreadLaunchRef>>
      keyed;
  keyed.reserve(roots.size());
  for (::dataflow::RootThreadLaunchRef root : roots) {
    auto resolved = dataflow.resolve(root);
    if (!resolved)
      return resolved.takeError();
    auto bytes = ::dataflow::encodeDataflowReference(dataflow.identity(), root);
    if (!bytes)
      return bytes.takeError();
    keyed.emplace_back(std::move(*bytes), root);
  }
  llvm::sort(keyed, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  for (std::size_t index = 1; index < keyed.size(); ++index)
    if (keyed[index - 1].first == keyed[index].first)
      return invalid("root launch set contains a duplicate");
  std::vector<::dataflow::RootThreadLaunchRef> result;
  result.reserve(keyed.size());
  for (auto &entry : keyed)
    result.push_back(entry.second);
  return result;
}

llvm::Expected<std::vector<CanonicalSystemPartitionBinding>>
canonicalizeAndValidateSystemPartition(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots,
    const SystemBindingPartitionPlan &plan) {
  auto canonicalRoots = canonicalRootThreadLaunchSet(dataflow, roots);
  if (!canonicalRoots)
    return canonicalRoots.takeError();
  auto expected = collectExpectedBindings(dataflow, *canonicalRoots);
  if (!expected)
    return expected.takeError();

  std::map<std::vector<std::uint8_t>, const ExpectedBinding *> expectedByKey;
  for (const ExpectedBinding &binding : *expected)
    expectedByKey.emplace(binding.canonicalKey, &binding);

  std::map<std::vector<std::uint8_t>, CanonicalSystemPartitionBinding>
      canonical;
  for (const SystemPresburgerBindingPartition &binding : plan.bindings) {
    auto key = canonicalBindingKeyBytes(binding.key, dataflow.identity());
    if (!key)
      return key.takeError();
    auto expectedIt = expectedByKey.find(*key);
    if (expectedIt == expectedByKey.end())
      return invalid(
          "partition plan contains a foreign or unreachable binding");
    if (binding.cells.empty())
      return invalid("Presburger partition contains no cells");
    if (canonical.find(*key) != canonical.end())
      return invalid("partition plan contains a duplicate binding");

    CanonicalSystemPartitionBinding normalized{binding.key, {}};
    normalized.cells.reserve(binding.cells.size());
    for (const SystemPresburgerCell &cell : binding.cells) {
      if (cell.dimensionCount !=
              expectedIt->second->legalDomain.dimensionCount ||
          cell.symbolCount != expectedIt->second->legalDomain.symbolCount)
        return invalid("Presburger cell has a foreign logical signature");
      auto normalizedCell = canonicalizeCell(cell);
      if (!normalizedCell)
        return normalizedCell.takeError();
      normalized.cells.push_back(std::move(*normalizedCell));
    }
    llvm::sort(normalized.cells, [](const SystemPresburgerCell &lhs,
                                    const SystemPresburgerCell &rhs) {
      return std::tie(lhs.dimensionCount, lhs.symbolCount, lhs.equalities,
                      lhs.inequalities) <
             std::tie(rhs.dimensionCount, rhs.symbolCount, rhs.equalities,
                      rhs.inequalities);
    });
    if (llvm::Error error = validatePartition(normalized.cells,
                                              expectedIt->second->legalDomain))
      return std::move(error);
    canonical.emplace(std::move(*key), std::move(normalized));
  }
  if (canonical.size() != expected->size())
    return invalid("partition plan omits a required binding");

  std::vector<CanonicalSystemPartitionBinding> result;
  result.reserve(canonical.size());
  for (auto &entry : canonical)
    result.push_back(std::move(entry.second));
  return result;
}

} // namespace detail

llvm::Expected<SystemBindingPartitionPlan>
projectWholeDomainPresburgerPartitionPlan(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> rootThreadLaunches) {
  auto roots =
      detail::canonicalRootThreadLaunchSet(dataflow, rootThreadLaunches);
  if (!roots)
    return roots.takeError();
  auto expected = collectExpectedBindings(dataflow, *roots);
  if (!expected)
    return expected.takeError();
  SystemBindingPartitionPlan plan;
  plan.bindings.reserve(expected->size());
  for (ExpectedBinding &binding : *expected)
    plan.bindings.push_back(
        {std::move(binding.key), {std::move(binding.legalDomain)}});
  return plan;
}

} // namespace loom::pnr
