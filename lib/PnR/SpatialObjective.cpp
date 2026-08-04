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

llvm::Error invalid(llvm::StringRef detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "dse_objective_invalid: selected Spatial %s reference is out of range",
      detail.str().c_str());
}

} // namespace

llvm::Expected<SpatialObjectiveProgram>
SpatialObjectiveProgram::get(const ResolvedObjectiveCatalogs &catalogs,
                             const ResolvedPnrObjectiveSelection &selection) {
  auto program = dse::ObjectiveProgram::get(catalogs);
  if (!program)
    return program.takeError();
  if (selection.selectedTotalOrdering >= program->totalOrderingCount())
    return invalid("total ordering");
  if (selection.selectedSearchEnergy >= program->weightedLevelCount())
    return invalid("search energy");

  std::uint64_t selectedViolations = 0;
  std::uint64_t selectedMeasures = 0;
  const auto violations = mappingViolationDescriptors();
  for (const ResolvedObjectiveDimension &dimension : catalogs.dimensions) {
    if (const auto *source =
            std::get_if<ResolvedMappingViolationObjectiveSource>(
                &dimension.source)) {
      const auto kind = source->kind;
      const std::uint32_t ordinal = static_cast<std::uint32_t>(kind);
      if (!spatialMappingViolationAvailable(kind))
        return unavailable(violations[ordinal].spelling);
      selectedViolations |= UINT64_C(1) << ordinal;
      continue;
    }
    if (const auto *source = std::get_if<ResolvedMappingMeasureObjectiveSource>(
            &dimension.source)) {
      selectedMeasures |= UINT64_C(1) << source->ordinal;
      continue;
    }
    return unavailable("Evaluation metric interaction");
  }
  return SpatialObjectiveProgram(
      std::move(*program), selectedViolations, selectedMeasures,
      selection.selectedTotalOrdering, selection.selectedSearchEnergy);
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
  if (llvm::Error error = program_.evaluate({violations, measures, {}}, result))
    return std::move(error);
  return result;
}

llvm::Expected<dse::ObjectiveWideValue> SpatialObjectiveProgram::selectedEnergy(
    const dse::ObjectiveVector &vector) const {
  return program_.weightedLevelValue(vector, selectedSearchEnergy_);
}

llvm::Expected<dse::ObjectiveSignedDifference>
SpatialObjectiveProgram::selectedEnergyDifference(
    const dse::ObjectiveVector &left, const dse::ObjectiveVector &right) const {
  return program_.signedWeightedLevelDifference(left, right,
                                                selectedSearchEnergy_);
}

llvm::Expected<int> SpatialObjectiveProgram::compareSelectedRank(
    const dse::ObjectiveVector &left,
    llvm::ArrayRef<std::uint8_t> leftCandidateKey,
    const dse::ObjectiveVector &right,
    llvm::ArrayRef<std::uint8_t> rightCandidateKey) const {
  return program_.compareTotalOrdering(
      left, leftCandidateKey, right, rightCandidateKey, selectedTotalOrdering_);
}
