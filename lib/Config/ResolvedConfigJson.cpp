#include "Config/ResolvedConfig.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactText.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <limits>
#include <string>
#include <variant>
#include <vector>

namespace {

llvm::StringRef violationName(loom::ResolvedPnrViolationKind violation) {
  using Kind = loom::ResolvedPnrViolationKind;
  switch (violation) {
#define LOOM_MAPPING_VIOLATION(Name, Ordinal, DisplayName, ConfigSpelling)     \
  case Kind::Name:                                                             \
    return ConfigSpelling;
#include "Common/MappingObjectiveKinds.def"
  }
  llvm_unreachable("all PnR violation kinds are handled");
}

llvm::json::Object ratioJson(const loom::ResolvedExactRatio &ratio) {
  return llvm::json::Object{{"numerator", ratio.numerator},
                            {"denominator", ratio.denominator}};
}

llvm::json::Object
routingNegotiationJson(const loom::ResolvedRoutingNegotiationPolicy &policy) {
  if (const auto *pathFinder =
          std::get_if<loom::ResolvedPathFinderPolicy>(&policy)) {
    return llvm::json::Object{
        {"kind", "pathfinder"},
        {"price_kernel",
         pathFinder->priceKernel ==
                 loom::ResolvedPathFinderPriceKernel::Multiplicative
             ? "multiplicative"
             : "additive"},
        {"present_pressure_initial", pathFinder->presentPressureInitial},
        {"present_pressure_growth",
         ratioJson(pathFinder->presentPressureGrowth)},
        {"history_pressure_increment", pathFinder->historyPressureIncrement}};
  }

  const auto &dual = std::get<loom::ResolvedDualSubgradientPolicy>(policy);
  llvm::json::Object direction;
  switch (dual.directionKernel) {
  case loom::ResolvedDualDirectionKernel::ProjectedSigned:
    direction = llvm::json::Object{{"kind", "projected_signed"}};
    break;
  case loom::ResolvedDualDirectionKernel::PositiveViolationOnly:
    direction = llvm::json::Object{{"kind", "positive_violation_only"}};
    break;
  case loom::ResolvedDualDirectionKernel::MomentumDeflected:
    direction = llvm::json::Object{{"kind", "momentum_deflected"},
                                   {"beta", ratioJson(*dual.momentum)}};
    break;
  }

  const loom::ResolvedDualStepSchedule &schedule = dual.stepSchedule;
  llvm::json::Object scheduleJson;
  switch (schedule.kind) {
  case loom::ResolvedDualStepScheduleKind::Constant:
    scheduleJson =
        llvm::json::Object{{"kind", "constant"}, {"step", schedule.first}};
    break;
  case loom::ResolvedDualStepScheduleKind::GeometricDecay:
    scheduleJson = llvm::json::Object{
        {"kind", "geometric_decay"},
        {"initial_step", schedule.first},
        {"minimum_step", schedule.second},
        {"decay", ratioJson({schedule.third, schedule.fourth})}};
    break;
  case loom::ResolvedDualStepScheduleKind::HarmonicDecay:
    scheduleJson = llvm::json::Object{{"kind", "harmonic_decay"},
                                      {"numerator", schedule.first},
                                      {"offset", schedule.second},
                                      {"minimum_step", schedule.third}};
    break;
  }
  return llvm::json::Object{{"kind", "dual_subgradient"},
                            {"direction_kernel", std::move(direction)},
                            {"step_schedule", std::move(scheduleJson)}};
}

llvm::json::Object pnrPolicyJson(const loom::ResolvedPnrPolicyConfig &policy) {
  const loom::ResolvedPnrSearchPolicy &search = policy.search;
  llvm::json::Array temporaryViolations;
  for (loom::ResolvedPnrViolationKind violation :
       policy.temporaryViolations.admitted)
    temporaryViolations.push_back(violationName(violation));

  llvm::json::Array focusedDimensions;
  for (std::uint32_t dimension :
       policy.objectiveSelection.focusedClosureDimensions)
    focusedDimensions.push_back(dimension);

  llvm::json::Array evaluationBindings;
  for (const loom::ResolvedPnrEvaluationBindingSelection &binding :
       policy.evaluationBindings)
    evaluationBindings.push_back(
        llvm::json::Object{{"obligation_template", binding.obligationTemplate},
                           {"interaction_domain", binding.interactionDomain}});

  llvm::json::Object exactRepair;
  if (search.exactRepair.kind == loom::ResolvedPnrExactRepairKind::Disabled) {
    exactRepair = llvm::json::Object{{"kind", "disabled"}};
  } else {
    exactRepair = llvm::json::Object{
        {"kind", "cp_sat"},
        {"max_region_decisions", search.exactRepair.maxRegionDecisions},
        {"max_solver_calls", search.exactRepair.maxSolverCalls}};
  }

  return llvm::json::Object{
      {"search_policy",
       llvm::json::Object{
           {"initializer",
            llvm::json::Object{
                {"seed_attempt_count", search.initializer.seedAttemptCount},
                {"assignment_attempt_limit_per_seed",
                 search.initializer.assignmentAttemptLimitPerSeed}}},
           {"action_proposal",
            llvm::json::Object{
                {"realization_binding_weight",
                 search.actionProposal.realizationBindingWeight},
                {"transport_routing_weight",
                 search.actionProposal.transportRoutingWeight},
                {"resource_allocation_weight",
                 search.actionProposal.resourceAllocationWeight}}},
           {"routing",
            llvm::json::Object{
                {"endpoint_expansion_limit",
                 search.routing.endpointExpansionLimit},
                {"negotiation_iteration_limit",
                 search.routing.negotiationIterationLimit},
                {"negotiation_policy",
                 routingNegotiationJson(search.routing.negotiation)},
                {"route_guidance_binding",
                 search.routing.routeGuidanceBinding
                     ? llvm::json::Value(*search.routing.routeGuidanceBinding)
                     : llvm::json::Value(nullptr)}}},
           {"annealing",
            llvm::json::Object{
                {"calibration_proposal_count",
                 search.annealing.calibrationProposalCount},
                {"positive_delta_quantile",
                 ratioJson(search.annealing.positiveDeltaQuantile)},
                {"target_initial_acceptance",
                 ratioJson(search.annealing.targetInitialAcceptance)},
                {"fallback_temperature", search.annealing.fallbackTemperature},
                {"minimum_temperature", search.annealing.minimumTemperature},
                {"cooling_ratio", ratioJson(search.annealing.coolingRatio)},
                {"proposals_per_level_base",
                 search.annealing.proposalsPerLevelBase},
                {"proposals_per_movable_decision",
                 search.annealing.proposalsPerMovableDecision}}},
           {"focused_closure",
            llvm::json::Object{
                {"proposal_limit", search.focusedClosureProposalLimit}}},
           {"exact_repair", std::move(exactRepair)}}},
      {"determinism_policy",
       llvm::json::Object{
           {"master_seed", policy.determinism.masterSeed},
           {"prng_protocol", "sha256_seeded_xoshiro256starstar_1_0"},
           {"acceptance_protocol", "exp_negative_q64_table_1_0"}}},
      {"temporary_violation_policy", std::move(temporaryViolations)},
      {"selected_total_ordering",
       policy.objectiveSelection.selectedTotalOrdering},
      {"selected_search_energy",
       policy.objectiveSelection.selectedSearchEnergy},
      {"focused_closure_dimensions", std::move(focusedDimensions)},
      {"evaluation_interaction_bindings", std::move(evaluationBindings)}};
}

llvm::StringRef
objectiveSourceName(const loom::ResolvedObjectiveScalarSource &source) {
  if (std::holds_alternative<loom::ResolvedMappingViolationObjectiveSource>(
          source))
    return "mapping_violation";
  if (std::holds_alternative<loom::ResolvedMappingMeasureObjectiveSource>(
          source))
    return "mapping_measure";
  return "evaluation_metric";
}

llvm::StringRef
objectiveDirectionName(loom::ResolvedObjectiveDirection direction) {
  return direction == loom::ResolvedObjectiveDirection::Minimize ? "minimize"
                                                                 : "maximize";
}

llvm::json::Value
objectiveScalarJson(const loom::ResolvedObjectiveScalar &value) {
  if (const auto *integer =
          std::get_if<loom::ResolvedObjectiveInteger>(&value)) {
    if (!integer->negative)
      return llvm::json::Value(integer->magnitude);
    if (integer->magnitude == (UINT64_C(1) << 63))
      return llvm::json::Value(std::numeric_limits<std::int64_t>::min());
    return llvm::json::Value(-static_cast<std::int64_t>(integer->magnitude));
  }
  const auto &decimal = std::get<loom::ResolvedObjectiveDecimal>(value);
  return llvm::json::Value(
      llvm::json::Object{{"coefficient", decimal.coefficient},
                         {"base10_exponent", decimal.base10Exponent}});
}

llvm::json::Object objectiveCatalogsJson(const loom::ResolvedDseConfig &dse) {
  llvm::json::Array authorizations;
  for (const loom::dse::ModelAuthorization &authorization :
       dse.modelAuthorizations) {
    const loom::SchemaVersion version =
        authorization.descriptor.schemaVersion();
    authorizations.push_back(llvm::json::Object{
        {"schema_major", version.major},
        {"schema_minor", version.minor},
        {"model_kind", authorization.descriptor.modelKind().ordinal()}});
  }

  llvm::json::Array templates;
  for (const loom::dse::EvidenceObligationTemplate &obligation :
       dse.evidenceObligationTemplates)
    templates.push_back(
        loom::formatArtifactLocalPayloadHex(obligation.canonicalBytes()));

  llvm::json::Array gates;
  for (const loom::dse::QualityGatePolicy &gate : dse.qualityGatePolicies)
    gates.push_back(loom::formatArtifactLocalPayloadHex(
        loom::dse::canonicalQualityGatePolicyBytes(gate)));

  llvm::json::Array planNodes;
  for (const loom::dse::DsePlanNodeDefinition &node : dse.planNodes)
    planNodes.push_back(loom::formatArtifactLocalPayloadHex(
        loom::dse::canonicalDsePlanNodeBytes(node)));

  const loom::ResolvedObjectiveCatalogs &catalogs = dse.objectiveCatalogs;
  llvm::json::Array dimensions;
  for (const loom::ResolvedObjectiveDimension &dimension :
       catalogs.dimensions) {
    llvm::json::Object object{
        {"source_kind", objectiveSourceName(dimension.source)},
        {"direction", objectiveDirectionName(dimension.direction)},
        {"origin", objectiveScalarJson(dimension.origin)},
        {"quantum", objectiveScalarJson(dimension.quantum)},
        {"lower_index", dimension.lowerIndex},
        {"upper_index", dimension.upperIndex}};
    if (const auto *violation =
            std::get_if<loom::ResolvedMappingViolationObjectiveSource>(
                &dimension.source)) {
      object.insert(
          {"source_ordinal", static_cast<std::uint32_t>(violation->kind)});
    } else if (const auto *measure =
                   std::get_if<loom::ResolvedMappingMeasureObjectiveSource>(
                       &dimension.source)) {
      object.insert({"source_ordinal", measure->ordinal});
    } else {
      const auto &metric =
          std::get<loom::ResolvedEvaluationMetricObjectiveSource>(
              dimension.source);
      object.insert(
          {"evidence_obligation_template", metric.evidenceObligationTemplate});
      object.insert({"metric_request_ordinal", metric.metricRequestOrdinal});
    }
    dimensions.push_back(std::move(object));
  }

  llvm::json::Array levels;
  for (const loom::ResolvedWeightedObjectiveLevel &level :
       catalogs.weightedLevels) {
    llvm::json::Array terms;
    for (const loom::ResolvedWeightedObjectiveTerm &term : level.terms)
      terms.push_back(llvm::json::Object{{"dimension", term.dimension},
                                         {"weight", term.weight}});
    levels.push_back(llvm::json::Object{{"terms", std::move(terms)}});
  }

  llvm::json::Array orderings;
  for (const loom::ResolvedTotalOrdering &ordering : catalogs.totalOrderings) {
    llvm::json::Array levelRefs;
    for (std::uint32_t level : ordering.weightedLevels)
      levelRefs.push_back(level);
    orderings.push_back(
        llvm::json::Object{{"weighted_levels", std::move(levelRefs)}});
  }
  return llvm::json::Object{
      {"model_authorizations", std::move(authorizations)},
      {"evidence_obligation_templates", std::move(templates)},
      {"objective_dimensions", std::move(dimensions)},
      {"weighted_levels", std::move(levels)},
      {"total_orderings", std::move(orderings)},
      {"quality_gate_policies", std::move(gates)},
      {"resolved_plan_nodes", std::move(planNodes)}};
}

llvm::json::Object
resolvedConfigJsonObject(const loom::ResolvedConfig &config) {
  const loom::adg::BuiltinTargetScale &scale = config.hardwareTarget.parameters;
  llvm::json::Object evaluation;
  if (config.evaluation.cadenceVoltusStaticRail) {
    const auto &binding = *config.evaluation.cadenceVoltusStaticRail;
    llvm::json::Array members;
    for (const loom::external_tool::ExternalFileTreeMember &member :
         binding.powerGridLibraryMembers) {
      members.push_back(llvm::json::Object{
          {"relative_path", member.relativePath},
          {"sha256", loom::formatExternalFileFingerprint(member.fingerprint)}});
    }
    llvm::json::Array entrypoints;
    for (const std::string &entrypoint :
         binding.powerGridLibraryEntrypoints)
      entrypoints.push_back(entrypoint);
    evaluation.insert({"cadence_voltus_static_rail",
                       llvm::json::Object{{"stable_provider_build_identity",
                                           binding.stableProviderBuildIdentity},
                                          {"power_grid_library_members",
                                           std::move(members)},
                                          {"power_grid_library_entrypoints",
                                           std::move(entrypoints)}}});
  }
  return llvm::json::Object{
      {"hardware_target",
       llvm::json::Object{
           {"template_identity", config.hardwareTarget.templateIdentity},
           {"schema_major", config.hardwareTarget.schemaVersion.major},
           {"schema_minor", config.hardwareTarget.schemaVersion.minor},
           {"parameters",
            llvm::json::Object{
                {"acc_core_count", scale.accCoreCount},
                {"spatial_pe_count", scale.spatialPeCount},
                {"temporal_pe_count", scale.temporalPeCount},
                {"spatial_memory_count", scale.spatialMemoryCount},
                {"temporal_memory_count", scale.temporalMemoryCount},
                {"temporal_resident_contexts", scale.temporalResidentContexts},
                {"gateway_count", scale.gatewayCount},
                {"memory_capacity_bytes", scale.memoryCapacityBytes}}}}},
      {"dse",
       llvm::json::Object{
           {"structured_ownership",
            llvm::json::Object{
                {"scope_expansion_limit",
                 static_cast<int64_t>(
                     config.dse.structuredOwnership.scopeExpansionLimit)}}},
           {"schedule",
            llvm::json::Object{{"scope_expansion_limit",
                                static_cast<int64_t>(
                                    config.dse.schedule.scopeExpansionLimit)}}},
           {"memory_communication",
            llvm::json::Object{
                {"scope_expansion_limit",
                 static_cast<int64_t>(
                     config.dse.memoryCommunication.scopeExpansionLimit)}}},
           {"dataflow_rewrite",
            llvm::json::Object{
                {"scope_expansion_limit",
                 static_cast<int64_t>(
                     config.dse.dataflowRewrite.scopeExpansionLimit)}}},
           {"tech_mapping",
            llvm::json::Object{
                {"match_row_attempt_limit",
                 config.dse.techMapping.matchRowAttemptLimit},
                {"partial_cover_expansion_limit",
                 config.dse.techMapping.partialCoverExpansionLimit},
                {"candidate_publication_limit",
                 config.dse.techMapping.candidatePublicationLimit}}},
           {"evaluation_and_objective_catalogs",
            objectiveCatalogsJson(config.dse)},
           {"spatial_pnr", pnrPolicyJson(config.dse.spatialPnr)},
           {"system_pnr", pnrPolicyJson(config.dse.systemPnr)}}},
      {"evaluation", std::move(evaluation)}};
}

} // namespace

std::string
loom::canonicalResolvedConfigJson(const loom::ResolvedConfig &config) {
  return llvm::formatv("{0:2}",
                       llvm::json::Value(resolvedConfigJsonObject(config)))
      .str();
}

loom::CanonicalSemanticBytes
loom::canonicalResolvedConfigBytes(const loom::ResolvedConfig &config) {
  const std::string json = canonicalResolvedConfigJson(config);
  return CanonicalSemanticBytes(
      std::vector<std::uint8_t>(json.begin(), json.end()));
}

loom::ArtifactIdentity
loom::resolvedConfigIdentity(const loom::ResolvedConfig &config) {
  return finalizeArtifactIdentity(ResolvedConfig::artifactSchema,
                                  canonicalResolvedConfigBytes(config));
}
