#include "DSE/TechMappingHardwareFeedback.h"

#include "Fabric/Artifact/FabricArtifact.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <map>
#include <vector>

namespace loom::dse {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "tech_mapping_hardware_feedback_invalid: " +
                                     message);
}

} // namespace

llvm::Expected<std::vector<SpatialMicroarchitectureDecisionDomain>>
projectTechMappingComputeContextGrowthDomains(
    const mapping::TechMappingComputeContextHallDeficit &feedback,
    const fabric::FabricArtifactView &module) {
  if (module.rootKind() != fabric::FabricRootKind::Module)
    return invalid("hardware feedback target is not a Module");
  if (feedback.deficit() == 0)
    return invalid("compute-context feedback has no positive deficit");

  std::map<std::uint64_t, std::uint64_t> currentCapacityByPe;
  for (const auto &group : feedback.groups()) {
    for (const fabric::InstructionContextRef context :
         group.compatibleContexts) {
      const auto schedule = module.peSchedule(context.pe);
      if (!schedule)
        return invalid("compatible context has no PE schedule");
      if (*schedule != ::fabric::Schedule::Temporal)
        continue;
      const std::uint64_t current = module.peResidentContextCount(context.pe);
      if (current == 0 || current > std::numeric_limits<std::uint32_t>::max() -
                                        feedback.deficit())
        return invalid("Temporal PE context growth exceeds u32");
      auto [found, inserted] =
          currentCapacityByPe.emplace(context.pe.id(), current);
      if (!inserted && found->second != current)
        return invalid("Temporal PE has inconsistent context capacity");
    }
  }

  std::vector<SpatialMicroarchitectureDecisionDomain> domains;
  domains.reserve(currentCapacityByPe.size());
  for (const auto &[pe, current] : currentCapacityByPe) {
    std::vector<std::uint32_t> capacities{
        static_cast<std::uint32_t>(current + 1)};
    if (feedback.deficit() != 1)
      capacities.push_back(
          static_cast<std::uint32_t>(current + feedback.deficit()));
    domains.push_back(ResizeInstructionStoreDomain{
        fabric::FabricPeOccurrenceRef(pe), std::move(capacities)});
  }
  return domains;
}

} // namespace loom::dse
