#include "TechMappingCandidateDomain.h"

#include "Config/ResolvedConfig.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <sys/resource.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "tech mapping search test: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

loom::ArtifactIdentity identity() {
  return take(loom::ArtifactIdentity::fromBytes(
      std::vector<std::uint8_t>(loom::ArtifactIdentity::byteSize, 0)));
}

dataflow::ActorRef actor(const loom::ArtifactIdentity &owner,
                         std::uint64_t ordinal) {
  return dataflow::ActorRef{owner, dataflow::ActorId(ordinal)};
}

loom::mapping::detail::TechMatchRow
row(std::uint8_t key, std::vector<std::size_t> actors,
    std::vector<std::size_t> contexts = {}) {
  return loom::mapping::detail::TechMatchRow{
      {key},
      std::move(actors),
      loom::mapping::TechComputeRealizationView{},
      std::move(contexts),
      std::nullopt};
}

loom::mapping::detail::TechMatchRow memoryRow(
    std::uint8_t key, std::vector<std::size_t> actors, std::uint8_t resource,
    loom::mapping::SpatialMemoryExclusiveResourceKind resourceKind = loom::
        mapping::SpatialMemoryExclusiveResourceKind::SpatialOperationPort) {
  loom::mapping::SpatialMemoryOccurrenceDemandView demand;
  demand.schedule = resourceKind ==
                            loom::mapping::SpatialMemoryExclusiveResourceKind::
                                SpatialOperationPort
                        ? fabric::Schedule::Spatial
                        : fabric::Schedule::Temporal;
  demand.occurrences = {{loom::fabric::FabricMemoryOccurrenceRef(1), 0},
                        {loom::fabric::FabricMemoryOccurrenceRef(2), 0}};
  demand.exclusiveResources = {{resourceKind, {resource}}};
  return loom::mapping::detail::TechMatchRow{
      {key},
      std::move(actors),
      loom::mapping::TechMemoryRealizationView{},
      {},
      std::move(demand)};
}

void assignIndependentComputeContextSupply(
    loom::mapping::detail::TechMatchDomain &domain) {
  domain.computeContextValueCount = domain.rows.size();
  for (auto [ordinal, candidate] : llvm::enumerate(domain.rows))
    candidate.computeContextValues = {ordinal};
}

loom::mapping::ResolvedTechMappingConfigView
config(std::uint64_t expansionLimit, std::uint64_t publicationLimit) {
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.techMapping.partialCoverExpansionLimit = expansionLimit;
  resolved.dse.techMapping.candidatePublicationLimit = publicationLimit;
  return take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
}

std::vector<std::uint8_t>
coverKey(llvm::ArrayRef<const loom::mapping::detail::TechMatchRow *> cover) {
  std::vector<std::uint8_t> result;
  for (const auto *selected : cover)
    result.push_back(selected->key.front());
  return result;
}

void completedProductSurvivesExpansionLimit() {
  const loom::ArtifactIdentity owner = identity();
  loom::mapping::detail::TechMatchDomain domain;
  for (std::uint64_t ordinal = 0; ordinal < 4; ++ordinal)
    domain.actors.push_back(actor(owner, ordinal));
  domain.rows = {
      row(0, {2}),    row(1, {0}), row(2, {1}),
      row(3, {0, 1}), row(4, {3}), row(5, {2, 3}),
  };
  assignIndependentComputeContextSupply(domain);

  loom::mapping::TechMappingGenerationAccounting accounting;
  const auto result = loom::mapping::detail::searchTechMatchCovers(
      domain, config(5, 16), accounting);
  if (result.exhausted)
    fail("a reached expansion limit was reported as exhaustive");
  if (result.covers.size() != 2 ||
      coverKey(result.covers[0]) != std::vector<std::uint8_t>({0, 3, 4}) ||
      coverKey(result.covers[1]) != std::vector<std::uint8_t>({0, 1, 2, 4}))
    fail("a completed component product was discarded at the expansion limit");
}

void sealedCoversFollowFormalRank() {
  const loom::ArtifactIdentity owner = identity();
  loom::mapping::detail::TechMatchDomain domain;
  for (std::uint64_t ordinal = 0; ordinal < 5; ++ordinal)
    domain.actors.push_back(actor(owner, ordinal));
  domain.rows = {
      row(0, {0}),     row(1, {0, 1, 2, 3}), row(2, {0, 1}),  row(3, {0, 1, 2}),
      row(4, {4}),     row(5, {3}),          row(6, {2}),     row(7, {1}),
      row(8, {1, 2}),  row(9, {1, 3}),       row(10, {1, 3}), row(11, {1, 4}),
      row(12, {1, 4}), row(13, {1, 3, 4}),
  };
  assignIndependentComputeContextSupply(domain);

  loom::mapping::TechMappingGenerationAccounting accounting;
  const auto result = loom::mapping::detail::searchTechMatchCovers(
      domain, config(10, 3), accounting);
  if (result.exhausted || result.covers.size() != 3 ||
      result.covers[0].size() != 2 || result.covers[1].size() != 3 ||
      result.covers[2].size() != 4)
    fail("sealed covers did not retain row-count-first canonical rank");
  if (coverKey(result.covers[1]) != std::vector<std::uint8_t>({3, 4, 5}) ||
      coverKey(result.covers[2]) != std::vector<std::uint8_t>({2, 4, 5, 6}))
    fail("sealed covers did not use canonical row keys as the rank tie-break");
}

long peakRssKiB() {
  struct rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0)
    fail("getrusage failed");
  return usage.ru_maxrss;
}

void independentComponentFrontierIsCompact() {
  constexpr std::size_t actorCount = 10000;
  constexpr std::size_t publicationLimit = 2;
  const loom::ArtifactIdentity owner = identity();
  loom::mapping::detail::TechMatchDomain domain;
  domain.actors.reserve(actorCount);
  domain.rows.reserve(actorCount * 2);
  for (std::size_t ordinal = 0; ordinal < actorCount; ++ordinal) {
    domain.actors.push_back(actor(owner, ordinal));
    domain.rows.push_back(
        row(static_cast<std::uint8_t>((ordinal * 2) >> 8), {ordinal}));
    domain.rows.back().key.push_back(
        static_cast<std::uint8_t>((ordinal * 2) & 0xff));
    domain.rows.push_back(
        row(static_cast<std::uint8_t>(((ordinal * 2) + 1) >> 8), {ordinal}));
    domain.rows.back().key.push_back(
        static_cast<std::uint8_t>(((ordinal * 2) + 1) & 0xff));
  }
  assignIndependentComputeContextSupply(domain);

  const long rssBefore = peakRssKiB();
  const auto start = std::chrono::steady_clock::now();
  loom::mapping::TechMappingGenerationAccounting accounting;
  const auto result = loom::mapping::detail::searchTechMatchCovers(
      domain, config(actorCount * 2, publicationLimit), accounting);
  const auto elapsed = std::chrono::steady_clock::now() - start;
  const long rssDelta = peakRssKiB() - rssBefore;

  if (result.covers.size() != publicationLimit || result.exhausted)
    fail("independent component product did not return its canonical prefix");
  if (elapsed > std::chrono::seconds(10))
    fail("independent component product exceeded its runtime gate");
  if (rssDelta > 16 * 1024)
    fail("independent component product exceeded its incremental RSS gate");
}

void realizationCountLowerBoundIsAdmissible() {
  const loom::ArtifactIdentity owner = identity();
  loom::mapping::detail::TechMatchDomain domain;
  for (std::uint64_t ordinal = 0; ordinal < 4; ++ordinal)
    domain.actors.push_back(actor(owner, ordinal));
  domain.rows = {
      row(0, {0}), row(1, {0, 1}), row(2, {1}),
      row(3, {2}), row(4, {3}),    row(5, {1, 2, 3}),
  };
  assignIndependentComputeContextSupply(domain);

  loom::mapping::TechMappingGenerationAccounting accounting;
  const auto result = loom::mapping::detail::searchTechMatchCovers(
      domain, config(1024, 2), accounting);
  if (result.covers.size() != 2 || result.covers.front().size() != 2 ||
      coverKey(result.covers.front()) != std::vector<std::uint8_t>({0, 5}))
    fail("a realization-count estimate reordered a smaller exact cover");
}

void exactComputeContextSupplyShapesFrontier() {
  const loom::ArtifactIdentity owner = identity();
  loom::mapping::detail::TechMatchDomain domain;
  for (std::uint64_t ordinal = 0; ordinal < 4; ++ordinal)
    domain.actors.push_back(actor(owner, ordinal));
  domain.computeContextValueCount = 4;
  domain.rows = {
      row(0, {0}, {0, 1, 3}), row(1, {0}, {2}),       row(2, {1}, {0, 1, 3}),
      row(3, {2}, {0, 1, 3}), row(4, {3}, {0, 1, 3}),
  };

  loom::mapping::TechMappingGenerationAccounting accounting;
  const auto result = loom::mapping::detail::searchTechMatchCovers(
      domain, config(1024, 1), accounting);
  if (result.covers.size() != 1 || coverKey(result.covers.front()) !=
                                       std::vector<std::uint8_t>({1, 2, 3, 4}))
    fail("the frontier did not replace an overlapping broad root domain");
  if (accounting.computeContextRejectedChecks != 1 ||
      accounting.computeContextMatchingChecks <=
          accounting.computeContextRejectedChecks ||
      accounting.computeContextMatchingWork == 0)
    fail("compute-context matching work was not accounted exactly");
}

void exactMemoryOccurrenceSupplyShapesFrontier() {
  const loom::ArtifactIdentity owner = identity();
  loom::mapping::detail::TechMatchDomain domain;
  for (std::uint64_t ordinal = 0; ordinal < 3; ++ordinal)
    domain.actors.push_back(actor(owner, ordinal));
  domain.rows = {
      memoryRow(0, {0}, 7),
      memoryRow(1, {1}, 7),
      memoryRow(2, {2}, 7),
      memoryRow(3, {2}, 9),
  };

  loom::mapping::TechMappingGenerationAccounting accounting;
  const auto result = loom::mapping::detail::searchTechMatchCovers(
      domain, config(1024, 1), accounting);
  if (result.covers.size() != 1 ||
      coverKey(result.covers.front()) != std::vector<std::uint8_t>({0, 1, 3}))
    fail("the frontier retained an overcommitted Spatial memory port");
  if (accounting.memorySupplyRejectedChecks != 1 ||
      accounting.memorySupplyChecks <= accounting.memorySupplyRejectedChecks ||
      accounting.memorySupplyChecks != accounting.memorySupplyPartialChecks +
                                           accounting.memorySupplyFullChecks ||
      accounting.memorySupplyFullChecks == 0 ||
      accounting.memorySupplyExclusiveResourceRejections != 1 ||
      accounting.memorySupplySpatialPortRejections != 1 ||
      accounting.memorySupplyTemporalIngressRejections != 0 ||
      accounting.memorySupplyInternalConnectionRejections != 0 ||
      accounting.memorySupplySearchWork == 0) {
    llvm::errs() << "memory supply counters: checks="
                 << accounting.memorySupplyChecks
                 << " partial=" << accounting.memorySupplyPartialChecks
                 << " full=" << accounting.memorySupplyFullChecks
                 << " rejected=" << accounting.memorySupplyRejectedChecks
                 << " exclusive="
                 << accounting.memorySupplyExclusiveResourceRejections
                 << " spatial_port="
                 << accounting.memorySupplySpatialPortRejections
                 << " temporal_ingress="
                 << accounting.memorySupplyTemporalIngressRejections
                 << " internal_connection="
                 << accounting.memorySupplyInternalConnectionRejections
                 << " work=" << accounting.memorySupplySearchWork << '\n';
    fail("memory occurrence-supply work was not accounted exactly");
  }

  loom::mapping::SpatialMemoryOccurrenceDemandView ingressA;
  ingressA.schedule = fabric::Schedule::Temporal;
  ingressA.occurrences = {{loom::fabric::FabricMemoryOccurrenceRef(3), 4}};
  ingressA.exclusiveResources = {{
      loom::mapping::SpatialMemoryExclusiveResourceKind::
          TemporalExternalIngress,
      {1},
  }};
  ingressA.residentDemand = 1;
  auto ingressB = ingressA;
  const std::array<const loom::mapping::SpatialMemoryOccurrenceDemandView *, 2>
      ingressDemands = {&ingressA, &ingressB};
  const auto ingressSupply =
      take(loom::mapping::analyzeSpatialMemoryOccurrenceSupply(ingressDemands));
  if (ingressSupply.failure !=
          loom::mapping::SpatialMemoryOccurrenceSupplyFailureKind::
              ExclusiveResourceDeficit ||
      ingressSupply.failingResourceKind !=
          loom::mapping::SpatialMemoryExclusiveResourceKind::
              TemporalExternalIngress)
    fail("Temporal memory ingress exclusivity lost its typed failure");

  ingressB.exclusiveResources.front().key = {2};
  ingressA.residentDemand = 3;
  ingressB.residentDemand = 3;
  const auto capacitySupply =
      take(loom::mapping::analyzeSpatialMemoryOccurrenceSupply(ingressDemands));
  if (capacitySupply.failure !=
          loom::mapping::SpatialMemoryOccurrenceSupplyFailureKind::
              ResidentCapacityDeficit ||
      capacitySupply.failingResidentDemand != 6 ||
      capacitySupply.failingResidentCapacity != 4)
    fail("Temporal memory capacity lost its occurrence-global accounting");
}

void prospectiveSeedHasOneKeyedOutcome() {
  const loom::ArtifactIdentity owner = identity();
  const std::array<dataflow::ActorRef, 1> actors = {actor(owner, 0)};
  loom::mapping::TechMappingGenerationAccounting accounting;
  loom::mapping::detail::TechMatchRowCollector collector(actors, 2, accounting,
                                                         {});

  if (!take(collector.beginSeed({0x10})))
    fail("the first prospective seed did not enter the finite prefix");
  if (llvm::Error error =
          collector.reject(loom::mapping::detail::TechMatchSeedRejectionReason::
                               CapabilityInadmissible))
    fail(llvm::toString(std::move(error)));
  if (collector.rejectionCount(
          loom::mapping::detail::TechMatchSeedRejectionReason::
              CapabilityInadmissible) != 1)
    fail("a failed prospective seed did not produce one typed rejection");

  if (!take(collector.beginSeed({0x20})))
    fail("the admitted prospective seed did not enter the finite prefix");
  if (llvm::Error error =
          collector.admit(loom::mapping::TechComputeRealizationView{}, actors))
    fail(llvm::toString(std::move(error)));
  auto rows = take(collector.takeRows());
  if (rows.size() != 1 || rows.front().key != std::vector<std::uint8_t>{0x20})
    fail("row admission did not preserve the prospective seed key");
  if (accounting.matchRowAttempts != 2)
    fail("prospective seed outcomes did not match attempt accounting");
}

void canonicalRejectedRangePreservesLimitAccounting() {
  const loom::ArtifactIdentity owner = identity();
  const std::array<dataflow::ActorRef, 1> actors = {actor(owner, 0)};
  loom::mapping::TechMappingGenerationAccounting accounting;
  loom::mapping::detail::TechMatchRowCollector collector(actors, 4, accounting,
                                                         {});

  if (!take(collector.beginSeed({0x10})))
    fail("the seed before a rejected range did not enter the prefix");
  if (llvm::Error error =
          collector.reject(loom::mapping::detail::TechMatchSeedRejectionReason::
                               CapabilityInadmissible))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = collector.rejectCanonicalSeedRange(
          {0x20}, {0x60}, 5, false,
          loom::mapping::detail::TechMatchSeedRejectionReason::
              RealizationInadmissible))
    fail(llvm::toString(std::move(error)));
  if (!collector.truncated() || accounting.matchRowAttempts != 4 ||
      collector.rejectionCount(
          loom::mapping::detail::TechMatchSeedRejectionReason::
              RealizationInadmissible) != 3)
    fail("canonical rejected range did not stop at the semantic limit");
  if (!take(collector.takeRows()).empty())
    fail("a rejected canonical range produced a match row");
}

} // namespace

int main() {
  independentComponentFrontierIsCompact();
  realizationCountLowerBoundIsAdmissible();
  exactComputeContextSupplyShapesFrontier();
  exactMemoryOccurrenceSupplyShapesFrontier();
  completedProductSurvivesExpansionLimit();
  sealedCoversFollowFormalRank();
  prospectiveSeedHasOneKeyedOutcome();
  canonicalRejectedRangePreservesLimitAccounting();
  llvm::outs() << "tech mapping search tests passed\n";
  return 0;
}
