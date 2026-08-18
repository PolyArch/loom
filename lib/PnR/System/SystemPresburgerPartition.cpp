#include "SystemPnrSearchDomainInternal.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace loom::pnr {
namespace {

using ::loom::mapping::SystemPresburgerCell;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_pnr_search_domain_invalid: " +
                                     message);
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
  const std::size_t rowWidth = static_cast<std::size_t>(cell.dimensionCount) +
                               cell.symbolCount + cell.localCount + 1;
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
  return ::loom::mapping::canonicalizeSystemPresburgerCell(cell);
}

llvm::Expected<std::vector<SystemPresburgerCell>>
cyclicCells(const SystemPresburgerCell &domain, std::size_t partitionCount) {
  if (partitionCount == 0)
    return invalid("cyclic partition count is zero");
  if (partitionCount == 1 || domain.dimensionCount == 0)
    return std::vector<SystemPresburgerCell>{domain};
  if (partitionCount >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
    return invalid("cyclic partition count exceeds i64");

  std::vector<SystemPresburgerCell> cells;
  cells.reserve(partitionCount);
  for (std::size_t residue = 0; residue < partitionCount; ++residue) {
    SystemPresburgerCell cell = domain;
    const std::size_t localColumn =
        static_cast<std::size_t>(cell.dimensionCount) + cell.symbolCount +
        cell.localCount;
    ++cell.localCount;
    for (auto &row : cell.equalities)
      row.insert(row.begin() + localColumn, 0);
    for (auto &row : cell.inequalities)
      row.insert(row.begin() + localColumn, 0);
    std::vector<std::int64_t> congruence(
        static_cast<std::size_t>(cell.dimensionCount) + cell.symbolCount +
            cell.localCount + 1,
        0);
    congruence[cell.dimensionCount - 1] = 1;
    congruence[localColumn] = -static_cast<std::int64_t>(partitionCount);
    congruence.back() = -static_cast<std::int64_t>(residue);
    cell.equalities.push_back(std::move(congruence));
    auto canonical = ::loom::mapping::canonicalizeSystemPresburgerCell(cell);
    if (!canonical)
      return canonical.takeError();
    cells.push_back(std::move(*canonical));
  }
  return cells;
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

} // namespace

namespace detail {

llvm::Expected<bool>
systemPresburgerCellsIntersect(const SystemPresburgerCell &lhs,
                               const SystemPresburgerCell &rhs) {
  return ::loom::mapping::systemPresburgerCellsIntersect(lhs, rhs);
}

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
      auto normalizedCell =
          ::loom::mapping::canonicalizeSystemPresburgerCell(cell);
      if (!normalizedCell)
        return normalizedCell.takeError();
      normalized.cells.push_back(std::move(*normalizedCell));
    }
    llvm::sort(normalized.cells, [](const SystemPresburgerCell &lhs,
                                    const SystemPresburgerCell &rhs) {
      return std::tie(lhs.dimensionCount, lhs.symbolCount, lhs.localCount,
                      lhs.equalities, lhs.inequalities) <
             std::tie(rhs.dimensionCount, rhs.symbolCount, rhs.localCount,
                      rhs.equalities, rhs.inequalities);
    });
    auto analysis = ::loom::mapping::analyzeSystemPresburgerPartition(
        normalized.cells, expectedIt->second->legalDomain);
    if (!analysis)
      return analysis.takeError();
    if (!analysis->liesWithinLegalDomain)
      return invalid("Presburger cell extends beyond the Dataflow may-domain");
    if (!analysis->cellsAreDisjoint)
      return invalid("Presburger partition cells overlap");
    if (!analysis->coversLegalDomain)
      return invalid(
          "Presburger partition does not cover the Dataflow may-domain");
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

llvm::Expected<SystemBindingPartitionPlan>
projectCyclicPresburgerPartitionPlan(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> rootThreadLaunches,
    std::size_t partitionCount) {
  auto roots =
      detail::canonicalRootThreadLaunchSet(dataflow, rootThreadLaunches);
  if (!roots)
    return roots.takeError();
  auto expected = collectExpectedBindings(dataflow, *roots);
  if (!expected)
    return expected.takeError();
  SystemBindingPartitionPlan plan;
  plan.bindings.reserve(expected->size());
  for (ExpectedBinding &binding : *expected) {
    auto cells = cyclicCells(binding.legalDomain, partitionCount);
    if (!cells)
      return cells.takeError();
    plan.bindings.push_back(
        {std::move(binding.key), std::move(*cells)});
  }
  auto canonical = detail::canonicalizeAndValidateSystemPartition(
      dataflow, *roots, plan);
  if (!canonical)
    return canonical.takeError();
  plan.bindings.clear();
  plan.bindings.reserve(canonical->size());
  for (detail::CanonicalSystemPartitionBinding &binding : *canonical)
    plan.bindings.push_back(
        {std::move(binding.key), std::move(binding.cells)});
  return plan;
}

} // namespace loom::pnr
