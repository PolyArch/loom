#include "SpatialTagPressureDiagnostic.h"

#include "Common/MappingDebugLog.h"
#include "Fabric/Identity/FabricRefText.h"
#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialRouteCostState.h"
#include "PnR/SpatialTagAssignment.h"

#include "llvm/Support/JSON.h"

#include <cstddef>
#include <utility>

namespace loom::pnr {

std::uint64_t reportSpatialTagDomainPressure(
    const SpatialCandidateState &candidate, const SpatialRouteCostState &costs,
    const SpatialTagAssignmentSummary &summary, std::uint64_t iteration,
    std::uint64_t sessionIteration) {
  const bool decision =
      loom::mapping_debug::enabled(loom::mapping_debug::Level::Decision);
  const bool detail =
      loom::mapping_debug::enabled(loom::mapping_debug::Level::Detail);
  const auto domains =
      candidate.problem().routing().tagContinuity().matchDomains();
  constexpr std::uint64_t decisionLimit = 16;
  constexpr std::size_t sampleLimit = 8;
  std::uint64_t pressureDomainCount = 0;
  std::uint64_t emittedCount = 0;
  for (PnrIndex domain = 0; domain < domains.size(); ++domain) {
    const std::uint64_t usage = summary.domainResidentCounts[domain];
    const std::uint64_t encodingCapacity =
        costs.tagDomainEncodingCapacity(domain);
    const std::uint64_t residentOveruse =
        costs.tagDomainResidentOveruse(domain);
    const std::uint64_t conflicts = summary.domainConflictCounts[domain];
    const std::uint64_t encodingOveruse =
        usage > encodingCapacity ? usage - encodingCapacity : 0;
    if (residentOveruse == 0 && encodingOveruse == 0 && conflicts == 0)
      continue;
    ++pressureDomainCount;
    if (!decision || (!detail && emittedCount >= decisionLimit))
      continue;
    ++emittedCount;

    std::uint64_t contributingNetCount = 0;
    llvm::json::Array netSample;
    for (PnrIndex logicalNet = 0;
         logicalNet < candidate.problem().transfers().logicalNets().size();
         ++logicalNet) {
      const std::size_t begin = summary.netDomainUseOffsets[logicalNet];
      const std::size_t end = summary.netDomainUseOffsets[logicalNet + 1];
      for (std::size_t incidence = begin; incidence < end; ++incidence) {
        if (summary.netDomainUseDomains[incidence] != domain)
          continue;
        ++contributingNetCount;
        if (detail || netSample.size() < sampleLimit) {
          llvm::json::Object row;
          row["logical_net"] = logicalNet;
          row["segment_count"] = summary.netDomainUseCounts[incidence];
          netSample.push_back(std::move(row));
        }
        break;
      }
    }
    loom::mapping_debug::emit(
        detail ? loom::mapping_debug::Level::Detail
               : loom::mapping_debug::Level::Decision,
        loom::mapping_debug::Stage::SpatialPnr,
        loom::mapping_debug::Event::TagDomainPressure,
        [&](llvm::json::Object &fields) {
          fields["iteration"] = iteration;
          fields["session_iteration"] = sessionIteration;
          fields["domain"] = domain;
          fields["kind"] = static_cast<std::uint64_t>(domains[domain].kind);
          fields["owner_ref"] =
              loom::fabric::printFabricRef(domains[domain].owner);
          if (domains[domain].ingress)
            fields["ingress_ref"] =
                loom::fabric::printFabricRef(*domains[domain].ingress);
          fields["tag_width_bits"] = domains[domain].tagWidthBits;
          fields["encoding_capacity"] = encodingCapacity;
          if (domains[domain].residentEntryCapacity)
            fields["resident_entry_capacity"] =
                *domains[domain].residentEntryCapacity;
          fields["resident_count"] = usage;
          fields["resident_overuse"] = residentOveruse;
          fields["encoding_overuse"] = encodingOveruse;
          fields["tag_conflicts"] = conflicts;
          fields["contributing_logical_net_count"] = contributingNetCount;
          fields[detail ? "logical_nets" : "logical_net_sample"] =
              std::move(netSample);
          fields["diagnostic_sample_limit"] = detail ? 0 : sampleLimit;
        });
  }
  return pressureDomainCount;
}

} // namespace loom::pnr
