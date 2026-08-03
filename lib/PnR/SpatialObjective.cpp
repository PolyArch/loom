#include "PnR/SpatialObjective.h"

#include "PnR/MappingObjective.h"
#include "PnR/SpatialCandidateState.h"

#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <system_error>
#include <utility>

using namespace loom;
using namespace loom::pnr;

namespace {

static_assert(resolvedPnrViolationKindCount <= 64);
static_assert(mappingMeasureKindCount <= 64);

llvm::Error unavailable(llvm::StringRef source) {
  return llvm::createStringError(
      std::make_error_code(std::errc::operation_not_supported),
      "objective_unavailable: required Spatial violation owner '%s' is "
      "absent",
      source.str().c_str());
}

} // namespace

llvm::Expected<SpatialObjectiveProgram>
SpatialObjectiveProgram::get(const ResolvedObjectiveCatalogs &catalogs) {
  auto program = dse::ObjectiveProgram::get(catalogs);
  if (!program)
    return program.takeError();

  std::uint64_t selectedViolations = 0;
  std::uint64_t selectedMeasures = 0;
  const auto violations = mappingViolationDescriptors();
  for (const ResolvedObjectiveDimension &dimension : catalogs.dimensions) {
    if (dimension.sourceKind == ResolvedObjectiveSourceKind::MappingViolation) {
      const auto kind =
          static_cast<ResolvedPnrViolationKind>(dimension.sourceOrdinal);
      if (!spatialMappingViolationAvailable(kind))
        return unavailable(violations[dimension.sourceOrdinal].spelling);
      selectedViolations |= UINT64_C(1) << dimension.sourceOrdinal;
      continue;
    }
    selectedMeasures |= UINT64_C(1) << dimension.sourceOrdinal;
  }
  return SpatialObjectiveProgram(std::move(*program), selectedViolations,
                                 selectedMeasures);
}

llvm::Expected<dse::ObjectiveVector> SpatialObjectiveProgram::evaluate(
    const SpatialCandidateState &candidate) const {
  std::array<std::uint64_t, resolvedPnrViolationKindCount> violations{};
  for (std::uint32_t ordinal = 0; ordinal != violations.size(); ++ordinal) {
    if ((selectedViolations_ & (UINT64_C(1) << ordinal)) == 0)
      continue;
    auto value = spatialMappingViolationValue(
        candidate, static_cast<ResolvedPnrViolationKind>(ordinal));
    if (!value)
      return value.takeError();
    violations[ordinal] = *value;
  }

  std::array<std::uint64_t, mappingMeasureKindCount> measures{};
  for (std::uint32_t ordinal = 0; ordinal != measures.size(); ++ordinal)
    if ((selectedMeasures_ & (UINT64_C(1) << ordinal)) != 0)
      measures[ordinal] = spatialMappingMeasureValue(
          candidate, static_cast<MappingMeasureKind>(ordinal));

  dse::ObjectiveVector result = program_.makeVector();
  if (llvm::Error error = program_.evaluate({violations, measures}, result))
    return std::move(error);
  return result;
}
