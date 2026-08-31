#include "PnR/PnrConfig.h"
#include "PnR/MappingObjective.h"
#include "PnR/RoutingNegotiation.h"

#include "Common/ComponentViewDigest.h"
#include "Config/ResolvedConfig.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "PnR config test: " << message << '\n';
  std::exit(1);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void requireRejected(llvm::Expected<T> value, llvm::StringRef fragment) {
  if (value)
    fail("expected rejection");
  const std::string message = llvm::toString(value.takeError());
  require(llvm::StringRef(message).contains(fragment), message);
}

void projectionAndAdoptionAreDomainTyped() {
  const loom::ResolvedConfig config = loom::defaultResolvedConfig();
  const loom::pnr::ResolvedPnrConfigView spatial =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(config));
  const loom::pnr::ResolvedPnrConfigView system =
      take(loom::pnr::projectResolvedSystemPnrConfigView(config));

  require(spatial.domain() == loom::pnr::PnrConfigDomain::Spatial,
          "Spatial projector returned the wrong domain");
  require(system.domain() == loom::pnr::PnrConfigDomain::System,
          "System projector returned the wrong domain");
  require(spatial.schemaDescriptorBytes() != system.schemaDescriptorBytes(),
          "Spatial and System descriptors are not distinct");
  require(llvm::StringRef(reinterpret_cast<const char *>(
                              spatial.schemaDescriptorBytes().data()),
                          spatial.schemaDescriptorBytes().size()) ==
              "loom.spatial_pnr.config.15.5",
          "Spatial PnR view has the wrong schema descriptor");
  require(llvm::StringRef(reinterpret_cast<const char *>(
                              system.schemaDescriptorBytes().data()),
                          system.schemaDescriptorBytes().size()) ==
              "loom.system_pnr.config.8.4",
          "System PnR view has the wrong schema descriptor");
  require(system.systemBindingPartitions().empty(),
          "base System PnR view invented a partition intent");
  require(spatial.digest() != system.digest(),
          "domain-distinct views have the same digest");

  const auto selectsRecurrence =
      [](const loom::pnr::ResolvedPnrConfigView &view) {
        return llvm::any_of(
            view.selectedObjectiveCatalogs().dimensions,
            [](const loom::ResolvedObjectiveDimension &dimension) {
              const auto *measure =
                  std::get_if<loom::ResolvedMappingMeasureObjectiveSource>(
                      &dimension.source);
              return measure &&
                     measure->ordinal ==
                         static_cast<std::uint32_t>(
                             loom::pnr::MappingMeasureKind::
                                 RecurrenceMinimumInitiationIntervalCycles);
            });
      };
  require(!selectsRecurrence(spatial),
          "Spatial objective selected System-owned recurrence timing");
  require(selectsRecurrence(system),
          "System objective omitted recurrence timing");

  const loom::pnr::ResolvedPnrConfigView adopted =
      take(loom::pnr::adoptResolvedSpatialPnrConfigView(
          spatial.schemaDescriptorBytes(), spatial.canonicalViewBytes(),
          spatial.digest()));
  require(adopted.canonicalViewBytes() == spatial.canonicalViewBytes(),
          "adoption changed canonical bytes");
  require(adopted.digest() == spatial.digest(),
          "adoption changed the component digest");

  requireRejected(loom::pnr::adoptResolvedSystemPnrConfigView(
                      spatial.schemaDescriptorBytes(),
                      spatial.canonicalViewBytes(), spatial.digest()),
                  "pnr_config_descriptor_mismatch");

  std::array<std::uint8_t, loom::ArtifactIdentity::byteSize> identityBytes{};
  identityBytes.fill(9);
  const auto dataflowIdentity =
      take(loom::ArtifactIdentity::fromBytes(identityBytes));
  const loom::pnr::SystemBindingPartitionIntent intent{
      {dataflowIdentity, dataflow::RootThreadLaunchId(7)}, 4};
  const auto specialized =
      take(loom::pnr::specializeResolvedSystemPnrConfigView(system, {intent}));
  require(specialized.digest() != system.digest() &&
              specialized.systemBindingPartitions().size() == 1 &&
              specialized.systemBindingPartitions().front() == intent,
          "System partition intent did not enter the exact config view");
  const auto adoptedSpecialized =
      take(loom::pnr::adoptResolvedSystemPnrConfigView(
          specialized.schemaDescriptorBytes(), specialized.canonicalViewBytes(),
          specialized.digest()));
  require(adoptedSpecialized.systemBindingPartitions() ==
              specialized.systemBindingPartitions(),
          "System partition intent did not survive canonical adoption");
  requireRejected(
      loom::pnr::specializeResolvedSystemPnrConfigView(spatial, {intent}),
      "requires a System PnR view");
}

void selectedAndUnselectedRecordsHaveExactDependencies() {
  const loom::ResolvedConfig base = loom::defaultResolvedConfig();
  const loom::pnr::ResolvedPnrConfigView baseView =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(base));

  loom::ResolvedConfig selectedChange = base;
  ++selectedChange.dse.spatialPnr.search.routing.endpointExpansionLimit;
  const loom::pnr::ResolvedPnrConfigView selectedView =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(selectedChange));
  require(selectedView.digest() != baseView.digest(),
          "selected policy change did not affect the view digest");

  loom::ResolvedConfig completionChange = base;
  completionChange.dse.spatialPnr.search.completionGoal =
      loom::ResolvedPnrCompletionGoal::FirstVerifiedCandidate;
  const loom::pnr::ResolvedPnrConfigView completionView =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(completionChange));
  require(completionView.digest() != baseView.digest() &&
              completionView.policy().search.completionGoal ==
                  loom::ResolvedPnrCompletionGoal::FirstVerifiedCandidate,
          "search completion goal did not enter the projected policy");

  loom::ResolvedConfig unselectedChange = base;
  auto &catalogs = unselectedChange.dse.objectiveCatalogs;
  catalogs.dimensions.push_back(
      {loom::ResolvedEvaluationMetricObjectiveSource{0, 0},
       loom::ResolvedObjectiveDirection::Maximize,
       loom::resolvedObjectiveDecimal(0, 0),
       loom::resolvedObjectiveDecimal(1, 0), 0, UINT64_MAX});
  const std::uint32_t unselectedDimension =
      static_cast<std::uint32_t>(catalogs.dimensions.size() - 1);
  catalogs.weightedLevels.insert(catalogs.weightedLevels.begin() + 6,
                                 {{{unselectedDimension, 1}}});
  catalogs.totalOrderings[0].weightedLevels = {9, 0, 4, 5, 7, 2, 1, 3};
  catalogs.totalOrderings[1].weightedLevels = {9, 0, 4, 5, 8, 2, 1, 3};
  unselectedChange.dse.spatialPnr.objectiveSelection.selectedSearchEnergy = 10;
  unselectedChange.dse.systemPnr.objectiveSelection.selectedSearchEnergy = 11;

  const loom::pnr::ResolvedPnrConfigView unselectedView =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(unselectedChange));
  require(unselectedView.digest() == baseView.digest(),
          "unselected catalog record changed the view digest");

  loom::ResolvedConfig stale = base;
  stale.dse.spatialPnr.objectiveSelection.selectedSearchEnergy = 99;
  requireRejected(loom::pnr::projectResolvedSpatialPnrConfigView(stale),
                  "resolved_pnr_policy_invalid");
}

void workBudgetIsDerivedFromTheSelectedPolicy() {
  loom::ResolvedConfig config = loom::defaultResolvedConfig();
  config.dse.spatialPnr.search.initializer.seedAttemptCount = 7;
  config.dse.spatialPnr.search.routing.endpointExpansionLimit = 123;
  config.dse.spatialPnr.search.routing.noProgressIterationLimit = 17;
  config.dse.spatialPnr.search.routing.noProgressTrendWindow = 5;
  config.dse.spatialPnr.search.annealing.temperatureLevelLimit = 19;
  config.dse.spatialPnr.search.exactRepair.maxSolverCalls = 456;
  const loom::pnr::ResolvedPnrConfigView view =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(config));
  const std::vector<loom::pnr::DeterministicWorkBudgetEntry> budgets =
      loom::pnr::deriveDeterministicWorkBudgetView(view);

  const auto find = [&](loom::pnr::PnrWorkUnit unit) {
    for (const auto &entry : budgets)
      if (entry.unit == unit)
        return entry.limit;
    fail("derived work budget omitted a policy owner");
  };
  require(find(loom::pnr::PnrWorkUnit::SeedAttempt) == 7,
          "seed-attempt budget was not derived");
  require(find(loom::pnr::PnrWorkUnit::EndpointExpansion) == 123,
          "endpoint-expansion budget was not derived");
  require(find(loom::pnr::PnrWorkUnit::ConsecutiveNoProgressIteration) == 17,
          "no-progress budget was not derived");
  require(find(loom::pnr::PnrWorkUnit::NoProgressTrendTransition) == 5,
          "no-progress trend window was not derived");
  require(find(loom::pnr::PnrWorkUnit::TemperatureLevel) == 19,
          "temperature-level budget was not derived");
  require(find(loom::pnr::PnrWorkUnit::ExactRepairSolverCall) == 456,
          "exact-repair budget was not derived");
}

void routingKernelsConsumeTheProjectedOwnerRecord() {
  const loom::pnr::ResolvedPnrConfigView view =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(
          loom::defaultResolvedConfig()));
  const auto *pathFinder = std::get_if<loom::ResolvedPathFinderPolicy>(
      &view.policy().search.routing.negotiation);
  require(pathFinder != nullptr, "default view did not select PathFinder");
  require(take(loom::pnr::pathFinderResourceCost(
              pathFinder->priceKernel, loom::pnr::routeCostScale,
              loom::pnr::routeCostScale, 1, 0)) ==
              2 * loom::pnr::routeCostScale,
          "routing kernel did not consume the projected price kernel");
  require(take(loom::pnr::ceilMulDiv(
              pathFinder->presentPressureInitial,
              pathFinder->presentPressureGrowth.numerator,
              pathFinder->presentPressureGrowth.denominator)) == 2,
          "routing kernel did not consume the projected growth ratio");
}

void mappingObjectiveRegistryIsClosedAndTyped() {
  const auto &registry = loom::pnr::mappingObjectiveRegistryDescriptor();
  require(registry.identity == "loom.mapping.pnr.objective" &&
              registry.schemaMajor == 3 && registry.schemaMinor == 3,
          "Mapping objective registry has the wrong identity");

  const auto violations = loom::pnr::mappingViolationDescriptors();
  require(
      violations.size() == 7 &&
          violations.front().kind ==
              loom::ResolvedPnrViolationKind::UnroutedObligation &&
          violations.back().kind ==
              loom::ResolvedPnrViolationKind::RuntimeCounterexampleViolation,
      "Mapping violation registry does not own the closed catalog");
  require(
      violations[1].kind == loom::ResolvedPnrViolationKind::CapacityOveruse &&
          violations[2].kind == loom::ResolvedPnrViolationKind::TagUnassigned &&
          violations[3].kind == loom::ResolvedPnrViolationKind::TagConflict,
      "Mapping violation registry changed the canonical catalog order");
  const auto builtin = loom::resolvedBuiltinSpatialPnrPolicy(
      loom::ResolvedProfilePreset::BalancedExplore);
  require(!llvm::is_contained(
              builtin.temporaryViolations.admitted,
              loom::ResolvedPnrViolationKind::RuntimeCounterexampleViolation),
          "runtime counterexamples became temporary search violations");

  const auto measures = loom::pnr::mappingMeasureDescriptors();
  require(measures.size() == 10 &&
              measures.front().kind ==
                  loom::pnr::MappingMeasureKind::TotalSelectedTraversalClaim &&
              measures.back().kind ==
                  loom::pnr::MappingMeasureKind::ProgressRouteAnchorCount,
          "Mapping measure registry does not own the closed catalog");

  const loom::ResolvedConfig config = loom::defaultResolvedConfig();
  const auto &catalogs = config.dse.objectiveCatalogs;
  const std::uint32_t pressureDimension =
      loom::resolvedPnrViolationKindCount +
      static_cast<std::uint32_t>(
          loom::pnr::MappingMeasureKind::SharedOperandIngressPressure);
  const std::uint32_t debtDimension = static_cast<std::uint32_t>(
      loom::ResolvedPnrViolationKind::ProgressProofDebt);
  const std::uint32_t shortfallDimension =
      loom::resolvedPnrViolationKindCount +
      static_cast<std::uint32_t>(
          loom::pnr::MappingMeasureKind::ProgressCapacityShortfall);
  const std::uint32_t anchorDimension =
      loom::resolvedPnrViolationKindCount +
      static_cast<std::uint32_t>(
          loom::pnr::MappingMeasureKind::ProgressRouteAnchorCount);
  require(catalogs.weightedLevels.size() == 11 &&
              catalogs.weightedLevels[3].terms.size() == 1 &&
              catalogs.weightedLevels[3].terms.front().dimension ==
                  pressureDimension &&
              catalogs.weightedLevels[3].terms.front().weight == 1 &&
              catalogs.weightedLevels[0].terms.front().dimension ==
                  debtDimension &&
              catalogs.weightedLevels[4].terms.front().dimension ==
                  shortfallDimension &&
              catalogs.weightedLevels[5].terms.front().dimension ==
                  anchorDimension &&
              catalogs.totalOrderings.size() == 2 &&
              llvm::equal(llvm::ArrayRef<std::uint32_t>(
                              catalogs.totalOrderings[0].weightedLevels)
                              .take_front(4),
                          std::array<std::uint32_t, 4>{8, 0, 4, 5}) &&
              catalogs.totalOrderings[0].weightedLevels.back() == 3 &&
              catalogs.totalOrderings[1].weightedLevels.back() == 3,
          "activity debt and operand pressure rank levels are not explicit");
}

void resolvedConfigUsesTheIndependentViolationCatalog() {
  require(loom::ResolvedConfig::artifactSchema.version.major == 11 &&
              loom::ResolvedConfig::artifactSchema.version.minor == 2,
          "ResolvedConfig has the wrong schema version");
  const std::string canonical =
      loom::canonicalResolvedConfigJson(loom::defaultResolvedConfig());
  require(!llvm::StringRef(canonical).contains("resource_time_overbooking") &&
              !llvm::StringRef(canonical).contains("buffer_overuse") &&
              !llvm::StringRef(canonical).contains(
                  "hard_service_contract_shortfall"),
          "ResolvedConfig retained a retired Mapping violation spelling");
}

void objectiveArithmeticIsPreflightedByThePnrView() {
  loom::ResolvedConfig config = loom::defaultResolvedConfig();
  auto &energy = config.dse.objectiveCatalogs.weightedLevels[6];
  energy.terms[0].weight = UINT64_MAX;
  energy.terms[1].weight = UINT64_MAX - 1;
  requireRejected(loom::pnr::projectResolvedSpatialPnrConfigView(config),
                  "weighted level domain overflows uint128");
}

void domainCapabilitiesFailClosed() {
  loom::ResolvedConfig config = loom::defaultResolvedConfig();
  config.dse.spatialPnr.search.routing.negotiation =
      loom::ResolvedDualSubgradientPolicy{
          loom::ResolvedDualDirectionKernel::PositiveViolationOnly,
          std::nullopt,
          {loom::ResolvedDualStepScheduleKind::Constant, 1, 0, 0, 0}};
  requireRejected(loom::pnr::projectResolvedSpatialPnrConfigView(config),
                  "Spatial PnR supports only PathFinder");

  config = loom::defaultResolvedConfig();
  config.dse.systemPnr.search.routing.negotiation =
      loom::ResolvedDualSubgradientPolicy{
          loom::ResolvedDualDirectionKernel::PositiveViolationOnly,
          std::nullopt,
          {loom::ResolvedDualStepScheduleKind::Constant, 1, 0, 0, 0}};
  (void)take(loom::pnr::projectResolvedSystemPnrConfigView(config));

  config = loom::defaultResolvedConfig();
  config.dse.systemPnr.search.exactRepair = {
      loom::ResolvedPnrExactRepairKind::CpSat, 8, 8};
  requireRejected(loom::pnr::projectResolvedSystemPnrConfigView(config),
                  "System PnR has no exact-repair provider");

  config = loom::defaultResolvedConfig();
  config.dse.objectiveCatalogs.dimensions.back().source =
      loom::ResolvedEvaluationMetricObjectiveSource{0, 0};
  config.dse.objectiveCatalogs.dimensions.back().origin =
      loom::resolvedObjectiveDecimal(0, 0);
  config.dse.objectiveCatalogs.dimensions.back().quantum =
      loom::resolvedObjectiveDecimal(1, 0);
  requireRejected(loom::pnr::projectResolvedSpatialPnrConfigView(config),
                  "unavailable Evaluation owner");
}

void malformedWireFailsClosed() {
  const loom::pnr::ResolvedPnrConfigView view =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(
          loom::defaultResolvedConfig()));
  std::vector<std::uint8_t> trailing(view.canonicalViewBytes().begin(),
                                     view.canonicalViewBytes().end());
  trailing.push_back(0);
  const loom::ComponentViewDigest trailingDigest = take(
      loom::computeComponentViewDigest(view.schemaDescriptorBytes(), trailing));
  requireRejected(loom::pnr::adoptResolvedSpatialPnrConfigView(
                      view.schemaDescriptorBytes(), trailing, trailingDigest),
                  "pnr_config_bytes_invalid");

  std::vector<std::uint8_t> staleDigestBytes(view.digest().bytes().begin(),
                                             view.digest().bytes().end());
  staleDigestBytes.back() ^= 1;
  const loom::ComponentViewDigest staleDigest =
      take(loom::ComponentViewDigest::fromBytes(staleDigestBytes));
  requireRejected(
      loom::pnr::adoptResolvedSpatialPnrConfigView(
          view.schemaDescriptorBytes(), view.canonicalViewBytes(), staleDigest),
      "component_view_digest_mismatch");
}

} // namespace

int main() {
  projectionAndAdoptionAreDomainTyped();
  selectedAndUnselectedRecordsHaveExactDependencies();
  workBudgetIsDerivedFromTheSelectedPolicy();
  routingKernelsConsumeTheProjectedOwnerRecord();
  mappingObjectiveRegistryIsClosedAndTyped();
  resolvedConfigUsesTheIndependentViolationCatalog();
  objectiveArithmeticIsPreflightedByThePnrView();
  domainCapabilitiesFailClosed();
  malformedWireFailsClosed();
  static_assert(
      !std::is_default_constructible_v<loom::pnr::ResolvedPnrConfigView>);
  llvm::outs() << "PnR config tests passed\n";
  return 0;
}
