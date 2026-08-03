#include "PnR/MappingObjective.h"

#include "PnR/SpatialCandidateState.h"

#include "llvm/Support/ErrorHandling.h"

#include <array>
#include <limits>
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

bool loom::pnr::spatialMappingViolationAvailable(
    ResolvedPnrViolationKind kind) {
  switch (kind) {
  case ResolvedPnrViolationKind::UnroutedObligation:
  case ResolvedPnrViolationKind::CapacityOveruse:
  case ResolvedPnrViolationKind::TagUnassigned:
  case ResolvedPnrViolationKind::TagConflict:
    return true;
  case ResolvedPnrViolationKind::ResourceTimeOverbooking:
  case ResolvedPnrViolationKind::BufferOveruse:
  case ResolvedPnrViolationKind::HardProgressViolation:
  case ResolvedPnrViolationKind::HardServiceContractShortfall:
    return false;
  }
  llvm_unreachable("unknown Mapping violation kind");
}

llvm::Expected<std::uint64_t>
loom::pnr::spatialMappingViolationValue(const SpatialCandidateState &candidate,
                                        ResolvedPnrViolationKind kind) {
  if (!spatialMappingViolationAvailable(kind)) {
    const auto ordinal = static_cast<std::uint32_t>(kind);
    if (ordinal >= violations.size())
      llvm_unreachable("unknown Mapping violation kind");
    return llvm::createStringError(
        std::make_error_code(std::errc::operation_not_supported),
        "objective_unavailable: required Spatial violation owner '%s' is "
        "absent",
        violations[ordinal].spelling.str().c_str());
  }
  switch (kind) {
  case ResolvedPnrViolationKind::UnroutedObligation:
    return candidate.unroutedObligationCount();
  case ResolvedPnrViolationKind::CapacityOveruse: {
    const std::uint64_t atomic = candidate.atomicCapacityOveruse();
    const std::uint64_t route = candidate.routeCapacityOveruse();
    if (route > std::numeric_limits<std::uint64_t>::max() - atomic)
      return llvm::createStringError(
          std::make_error_code(std::errc::value_too_large),
          "Spatial CapacityOveruse exceeds u64");
    return atomic + route;
  }
  case ResolvedPnrViolationKind::TagUnassigned:
    return candidate.tagUnassignedCount();
  case ResolvedPnrViolationKind::TagConflict:
    return candidate.tagConflictCount();
  case ResolvedPnrViolationKind::ResourceTimeOverbooking:
  case ResolvedPnrViolationKind::BufferOveruse:
  case ResolvedPnrViolationKind::HardProgressViolation:
  case ResolvedPnrViolationKind::HardServiceContractShortfall:
    llvm_unreachable("unavailable Spatial violation passed preflight");
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
