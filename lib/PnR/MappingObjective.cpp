#include "PnR/MappingObjective.h"

#include "PnR/SpatialCandidateState.h"

#include "llvm/Support/ErrorHandling.h"

#include <array>
#include <system_error>

using namespace loom;
using namespace loom::pnr;

namespace {

constexpr MappingObjectiveRegistryDescriptor registry{
    "loom.mapping.pnr.objective", 1, 0};

constexpr std::array<MappingViolationDescriptor, resolvedPnrViolationKindCount>
    violations{{
#define LOOM_MAPPING_VIOLATION(Name, Ordinal, DisplayName, ConfigSpelling)     \
  {ResolvedPnrViolationKind::Name, DisplayName},
#include "Common/MappingObjectiveKinds.def"
    }};

constexpr std::array<MappingMeasureDescriptor, mappingMeasureKindCount>
    measures{{
#define LOOM_MAPPING_MEASURE(Name, Ordinal, DisplayName)                       \
  {MappingMeasureKind::Name, DisplayName},
#include "Common/MappingObjectiveKinds.def"
    }};

} // namespace

const MappingObjectiveRegistryDescriptor &
loom::pnr::mappingObjectiveRegistryDescriptor() {
  return registry;
}

llvm::ArrayRef<MappingViolationDescriptor>
loom::pnr::mappingViolationDescriptors() {
  return violations;
}

llvm::ArrayRef<MappingMeasureDescriptor>
loom::pnr::mappingMeasureDescriptors() {
  return measures;
}

llvm::Expected<std::uint64_t>
loom::pnr::spatialMappingViolationValue(const SpatialCandidateState &candidate,
                                        ResolvedPnrViolationKind kind) {
  switch (kind) {
  case ResolvedPnrViolationKind::UnroutedObligation:
    return candidate.unroutedObligationCount();
  case ResolvedPnrViolationKind::CapacityOveruse:
    return candidate.capacityOveruse();
  case ResolvedPnrViolationKind::ResourceTimeOverbooking:
  case ResolvedPnrViolationKind::BufferOveruse:
  case ResolvedPnrViolationKind::TagUnassigned:
  case ResolvedPnrViolationKind::TagConflict:
  case ResolvedPnrViolationKind::HardProgressViolation:
  case ResolvedPnrViolationKind::HardServiceContractShortfall:
    return llvm::createStringError(
        std::make_error_code(std::errc::operation_not_supported),
        "objective_unavailable: required Spatial violation projection is "
        "absent");
  }
  llvm_unreachable("unknown Mapping violation kind");
}

std::uint64_t
loom::pnr::spatialMappingMeasureValue(const SpatialCandidateState &candidate,
                                      MappingMeasureKind kind) {
  switch (kind) {
  case MappingMeasureKind::TotalSelectedTraversalClaim:
    return candidate.totalSelectedTraversalClaim();
  }
  llvm_unreachable("unknown Mapping measure kind");
}
