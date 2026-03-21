#include "fcc/Mapper/MapperOptions.h"

#include <sstream>

namespace fcc {

namespace {

bool requirePositiveDouble(double value, const char *name, std::string &error) {
  if (value > 0.0)
    return true;
  error = std::string(name) + " must be > 0";
  return false;
}

bool requireNonNegativeDouble(double value, const char *name,
                              std::string &error) {
  if (value >= 0.0)
    return true;
  error = std::string(name) + " must be >= 0";
  return false;
}

bool requireDoubleInRange(double value, double minValue, double maxValue,
                          const char *name, std::string &error) {
  if (value >= minValue && value <= maxValue)
    return true;
  std::ostringstream oss;
  oss << name << " must be in [" << minValue << ", " << maxValue << "]";
  error = oss.str();
  return false;
}

bool requirePositiveUnsigned(unsigned value, const char *name,
                             std::string &error) {
  if (value > 0)
    return true;
  error = std::string(name) + " must be > 0";
  return false;
}

bool requireDisabledOrPositiveInt(int value, const char *name,
                                  std::string &error) {
  if (value == -1 || value > 0)
    return true;
  error = std::string(name) + " must be -1 or > 0";
  return false;
}

bool requireDisabledOrPositiveDouble(double value, const char *name,
                                     std::string &error) {
  if (value == -1.0 || value > 0.0)
    return true;
  error = std::string(name) + " must be -1 or > 0";
  return false;
}

bool requireMinUnsigned(unsigned value, unsigned minValue, const char *name,
                        std::string &error) {
  if (value >= minValue)
    return true;
  std::ostringstream oss;
  oss << name << " must be >= " << minValue;
  error = oss.str();
  return false;
}

} // namespace

bool validateMapperOptions(const MapperOptions &opts, std::string &error) {
  if (!requirePositiveDouble(opts.budgetSeconds, "budget_seconds", error) ||
      !requireDisabledOrPositiveDouble(opts.snapshotIntervalSeconds,
                                       "snapshot_interval_seconds", error) ||
      !requireDisabledOrPositiveInt(opts.snapshotIntervalRounds,
                                    "snapshot_interval_rounds", error) ||
      !requirePositiveUnsigned(opts.interleavedRounds,
                               "interleaved_rounds", error) ||
      !requirePositiveUnsigned(opts.cpSatGlobalNodeLimit,
                               "cpsat_global_node_limit", error) ||
      !requirePositiveUnsigned(opts.cpSatNeighborhoodNodeLimit,
                               "cpsat_neighborhood_node_limit", error) ||
      !requireNonNegativeDouble(opts.cpSatTimeLimitSeconds,
                                "cpsat_time_limit_seconds", error) ||
      !requirePositiveDouble(opts.routingHeuristicWeight,
                             "routing_heuristic_weight", error) ||
      !requirePositiveDouble(opts.congestionHistoryFactor,
                             "congestion_history_factor", error) ||
      !requirePositiveDouble(opts.congestionHistoryScale,
                             "congestion_history_scale", error) ||
      !requirePositiveDouble(opts.congestionPresentFactor,
                             "congestion_present_factor", error) ||
      !requireNonNegativeDouble(opts.congestionPlacementWeight,
                                "congestion_placement_weight", error) ||
      !requireNonNegativeDouble(opts.memorySharingPenalty,
                                "memory_sharing_penalty", error)) {
    return false;
  }
  if (opts.snapshotIntervalSeconds > 0.0 && opts.snapshotIntervalRounds > 0) {
    error = "snapshot_interval_seconds and snapshot_interval_rounds cannot both be enabled";
    return false;
  }

  if (!requirePositiveDouble(opts.refinement.initialTemperature,
                             "refinement.initial_temperature", error) ||
      !requirePositiveDouble(opts.refinement.coolingRate,
                             "refinement.cooling_rate", error) ||
      !requirePositiveUnsigned(opts.refinement.adaptiveWindow,
                               "refinement.adaptive_window", error) ||
      !requirePositiveUnsigned(opts.refinement.routeAwareNeighborhoodEdgeCap,
                               "refinement.route_aware_neighborhood_edge_cap",
                               error) ||
      !requirePositiveUnsigned(
          opts.refinement.routeAwareCheckpointAcceptedMoveBatch,
          "refinement.route_aware_checkpoint_accepted_move_batch",
                               error) ||
      !requireDoubleInRange(opts.refinement.targetAcceptanceLow, 0.0, 1.0,
                            "refinement.target_acceptance_low", error) ||
      !requireDoubleInRange(opts.refinement.targetAcceptanceHigh, 0.0, 1.0,
                            "refinement.target_acceptance_high", error) ||
      !requirePositiveDouble(
          opts.refinement.coldAcceptanceReheatMultiplier,
          "refinement.cold_acceptance_reheat_multiplier", error) ||
      !requireDoubleInRange(
          opts.refinement.hotAcceptanceCoolingMultiplier, 0.0, 1.0,
          "refinement.hot_acceptance_cooling_multiplier", error) ||
      !requirePositiveDouble(opts.refinement.plateauReheatMultiplier,
                             "refinement.plateau_reheat_multiplier",
                             error) ||
      !requirePositiveDouble(opts.refinement.maxTemperatureScale,
                             "refinement.max_temperature_scale", error) ||
      !requirePositiveDouble(opts.refinement.minTemperature,
                             "refinement.min_temperature", error) ||
      !requirePositiveUnsigned(opts.refinement.iterationsPerPlacedNode,
                               "refinement.iterations_per_placed_node",
                               error) ||
      !requirePositiveUnsigned(opts.refinement.iterationCap,
                               "refinement.iteration_cap", error) ||
      !requirePositiveDouble(opts.refinement.budgetFraction,
                             "refinement.budget_fraction", error) ||
      !requirePositiveUnsigned(opts.refinement.relocateTopCandidateLimit,
                               "refinement.relocate_top_candidate_limit",
                               error)) {
    return false;
  }
  if (opts.refinement.budgetFraction > 1.0) {
    error = "refinement.budget_fraction must be <= 1";
    return false;
  }
  if (opts.refinement.targetAcceptanceLow >
      opts.refinement.targetAcceptanceHigh) {
    error =
        "refinement.target_acceptance_low must be <= refinement.target_acceptance_high";
    return false;
  }
  if (opts.refinement.coldAcceptanceReheatMultiplier < 1.0) {
    error =
        "refinement.cold_acceptance_reheat_multiplier must be >= 1";
    return false;
  }
  if (opts.refinement.plateauReheatMultiplier < 1.0) {
    error = "refinement.plateau_reheat_multiplier must be >= 1";
    return false;
  }
  if (opts.refinement.maxTemperatureScale < 1.0) {
    error = "refinement.max_temperature_scale must be >= 1";
    return false;
  }

  if (!requirePositiveUnsigned(opts.lane.autoSerialNodeThreshold,
                               "lane.auto_serial_node_threshold", error) ||
      !requirePositiveUnsigned(opts.lane.autoLaneCap, "lane.auto_lane_cap",
                               error) ||
      !requireMinUnsigned(opts.lane.routingBeamWidth, 0,
                          "lane.routing_beam_width", error) ||
      !requireNonNegativeDouble(opts.lane.finalPolishReserveFraction,
                                "lane.final_polish_reserve_fraction", error) ||
      !requirePositiveUnsigned(opts.lane.laneSeedStride,
                               "lane.lane_seed_stride", error) ||
      !requirePositiveUnsigned(opts.lane.restartSeedStride,
                               "lane.restart_seed_stride", error) ||
      !requirePositiveDouble(opts.lane.globalCPSatMinTimeSingleLane,
                             "lane.global_cpsat_min_time_single_lane",
                             error) ||
      !requirePositiveDouble(opts.lane.globalCPSatMinTimeMultiLane,
                             "lane.global_cpsat_min_time_multi_lane", error) ||
      !requirePositiveDouble(opts.lane.boundaryCPSatMinTimeSingleLane,
                             "lane.boundary_cpsat_min_time_single_lane",
                             error) ||
      !requirePositiveDouble(opts.lane.boundaryCPSatMinTimeMultiLane,
                             "lane.boundary_cpsat_min_time_multi_lane",
                             error) ||
      !requirePositiveUnsigned(opts.lane.boundaryNeighborhoodCap,
                               "lane.boundary_neighborhood_cap", error) ||
      !requirePositiveUnsigned(opts.lane.unroutedDiagnosticLimit,
                               "lane.unrouted_diagnostic_limit", error)) {
    return false;
  }
  if (opts.lane.finalPolishReserveFraction >= 1.0) {
    error = "lane.final_polish_reserve_fraction must be < 1";
    return false;
  }

  if (!requirePositiveUnsigned(opts.routing.tightRipupMinEdges,
                               "routing.tight_ripup_min_edges", error) ||
      !requirePositiveUnsigned(opts.routing.tightRipupTargetCap,
                               "routing.tight_ripup_target_cap", error) ||
      !requirePositiveUnsigned(opts.routing.failedEdgeDiagnosticLimit,
                               "routing.failed_edge_diagnostic_limit", error) ||
      !requirePositiveUnsigned(opts.routing.moduleOutputGroupMaxEdges,
                               "routing.module_output_group_max_edges",
                               error) ||
      !requirePositiveUnsigned(opts.routing.moduleOutputFirstHopLimit,
                               "routing.module_output_first_hop_limit",
                               error) ||
      !requirePositiveUnsigned(opts.routing.moduleOutputBranchLimit,
                               "routing.module_output_branch_limit", error) ||
      !requirePositiveUnsigned(opts.routing.sourceFanoutFirstHopLimit,
                               "routing.source_fanout_first_hop_limit",
                               error) ||
      !requireNonNegativeDouble(opts.routing.ripupHistoryBump,
                                "routing.ripup_history_bump", error)) {
    return false;
  }

  if (!requirePositiveDouble(opts.congestion.saturationPenalty,
                             "congestion.saturation_penalty", error) ||
      !requirePositiveDouble(opts.congestion.historyIncrementCap,
                             "congestion.history_increment_cap", error) ||
      !requireDoubleInRange(opts.congestion.historyDecay, 0.0, 1.0,
                            "congestion.history_decay", error) ||
      !requireNonNegativeDouble(opts.congestion.routingOutputHistoryBump,
                                "congestion.routing_output_history_bump",
                                error) ||
      !requireDoubleInRange(opts.congestion.routingOutputHistoryDecay, 0.0,
                            1.0,
                            "congestion.routing_output_history_decay",
                            error)) {
    return false;
  }

  if (!requirePositiveDouble(opts.timing.recurrenceEdgeWeightMultiplier,
                             "timing.recurrence_edge_weight_multiplier",
                             error) ||
      !requireNonNegativeDouble(opts.timing.recurrenceNodeLatencyWeight,
                                "timing.recurrence_node_latency_weight",
                                error) ||
      !requireNonNegativeDouble(opts.timing.recurrenceNodeIntervalWeight,
                                "timing.recurrence_node_interval_weight",
                                error) ||
      !requirePositiveDouble(opts.timing.combinationalNodeDelay,
                             "timing.combinational_node_delay", error) ||
      !requirePositiveDouble(opts.timing.routingHopDelay,
                             "timing.routing_hop_delay", error)) {
    return false;
  }

  if (!requirePositiveUnsigned(opts.bufferization.maxIterations,
                               "bufferization.max_iterations", error) ||
      !requirePositiveUnsigned(opts.bufferization.outerJointIterations,
                               "bufferization.outer_joint_iterations", error) ||
      !requireNonNegativeDouble(opts.bufferization.minThroughputImprovement,
                                "bufferization.min_throughput_improvement",
                                error) ||
      !requireNonNegativeDouble(opts.bufferization.clockTieBreakImprovement,
                                "bufferization.clock_tie_break_improvement",
                                error)) {
    return false;
  }

  if (!requirePositiveUnsigned(opts.relaxedRouting.legalizationPasses,
                               "relaxed_routing.legalization_passes",
                               error) ||
      !requirePositiveDouble(opts.relaxedRouting.baseOverusePenalty,
                             "relaxed_routing.base_overuse_penalty", error) ||
      !requirePositiveDouble(opts.relaxedRouting.repeatedOveruseScale,
                             "relaxed_routing.repeated_overuse_scale",
                             error) ||
      !requirePositiveUnsigned(opts.relaxedRouting.rejectCheckpointOveruseCap,
                               "relaxed_routing.reject_checkpoint_overuse_cap",
                               error)) {
    return false;
  }
  if (opts.relaxedRouting.repeatedOveruseScale < 1.0) {
    error = "relaxed_routing.repeated_overuse_scale must be >= 1";
    return false;
  }

  if (!requirePositiveDouble(opts.cpSatTuning.placementCostScale,
                             "cpsat_tuning.placement_cost_scale", error) ||
      !requirePositiveUnsigned(opts.cpSatTuning.workerFallbackConcurrency,
                               "cpsat_tuning.worker_fallback_concurrency",
                               error) ||
      !requirePositiveUnsigned(opts.cpSatTuning.workerCap,
                               "cpsat_tuning.worker_cap", error) ||
      !requirePositiveUnsigned(opts.cpSatTuning.globalCandidateLimit,
                               "cpsat_tuning.global_candidate_limit", error) ||
      !requirePositiveUnsigned(opts.cpSatTuning.neighborhoodExpansionBase,
                               "cpsat_tuning.neighborhood_expansion_base",
                               error) ||
      !requirePositiveUnsigned(opts.cpSatTuning.neighborhoodExpansionCap,
                               "cpsat_tuning.neighborhood_expansion_cap",
                               error) ||
      !requirePositiveDouble(opts.cpSatTuning.veryTightFocusWeightFloor,
                             "cpsat_tuning.very_tight_focus_weight_floor",
                             error) ||
      !requirePositiveDouble(opts.cpSatTuning.veryTightFocusWeightScale,
                             "cpsat_tuning.very_tight_focus_weight_scale",
                             error) ||
      !requirePositiveDouble(opts.cpSatTuning.tightFocusWeightFloor,
                             "cpsat_tuning.tight_focus_weight_floor", error) ||
      !requirePositiveDouble(opts.cpSatTuning.tightFocusWeightScale,
                             "cpsat_tuning.tight_focus_weight_scale", error) ||
      !requirePositiveUnsigned(opts.cpSatTuning.veryTightCandidateLimit,
                               "cpsat_tuning.very_tight_candidate_limit",
                               error) ||
      !requirePositiveUnsigned(opts.cpSatTuning.tightCandidateLimit,
                               "cpsat_tuning.tight_candidate_limit", error) ||
      !requirePositiveUnsigned(opts.cpSatTuning.defaultCandidateLimit,
                               "cpsat_tuning.default_candidate_limit", error)) {
    return false;
  }

  const auto &exact = opts.localRepair.exact;
  if (!requirePositiveUnsigned(exact.neighborhoodPassMin,
                               "local_repair.exact.neighborhood_pass_min",
                               error) ||
      !requirePositiveUnsigned(exact.neighborhoodPassCap,
                               "local_repair.exact.neighborhood_pass_cap",
                               error) ||
      !requirePositiveDouble(exact.microDeadlineMs,
                             "local_repair.exact.micro_deadline_ms", error) ||
      !requirePositiveDouble(exact.tightDeadlineMs,
                             "local_repair.exact.tight_deadline_ms", error) ||
      !requirePositiveDouble(exact.mediumDeadlineMs,
                             "local_repair.exact.medium_deadline_ms", error) ||
      !requirePositiveDouble(exact.defaultDeadlineMs,
                             "local_repair.exact.default_deadline_ms", error) ||
      !requirePositiveDouble(exact.deadlineScale,
                             "local_repair.exact.deadline_scale", error) ||
      !requirePositiveDouble(exact.microDeadlineScale,
                             "local_repair.exact.micro_deadline_scale", error) ||
      !requirePositiveUnsigned(exact.microFirstHopLimit,
                               "local_repair.exact.micro_first_hop_limit",
                               error) ||
      !requirePositiveUnsigned(exact.tightFirstHopLimit,
                               "local_repair.exact.tight_first_hop_limit",
                               error) ||
      !requirePositiveUnsigned(exact.mediumFirstHopLimit,
                               "local_repair.exact.medium_first_hop_limit",
                               error) ||
      !requirePositiveUnsigned(exact.defaultFirstHopLimit,
                               "local_repair.exact.default_first_hop_limit",
                               error) ||
      !requirePositiveUnsigned(exact.microCandidatePathLimit,
                               "local_repair.exact.micro_candidate_path_limit",
                               error) ||
      !requirePositiveUnsigned(exact.tightCandidatePathLimit,
                               "local_repair.exact.tight_candidate_path_limit",
                               error) ||
      !requirePositiveUnsigned(exact.mediumCandidatePathLimit,
                               "local_repair.exact.medium_candidate_path_limit",
                               error) ||
      !requirePositiveUnsigned(exact.defaultCandidatePathLimit,
                               "local_repair.exact.default_candidate_path_limit",
                               error) ||
      !requireNonNegativeDouble(exact.localHistoryBump,
                                "local_repair.exact.local_history_bump",
                                error)) {
    return false;
  }
  if (!requireMinUnsigned(exact.neighborhoodPassCap, exact.neighborhoodPassMin,
                          "local_repair.exact.neighborhood_pass_cap", error)) {
    return false;
  }

  if (!requirePositiveDouble(opts.localRepair.earlyCPSatMinTime,
                             "local_repair.early_cpsat_min_time", error) ||
      !requirePositiveUnsigned(opts.localRepair.microRecursionDepthLimit,
                               "local_repair.micro_recursion_depth_limit",
                               error) ||
      !requirePositiveUnsigned(opts.localRepair.defaultRecursionDepthLimit,
                               "local_repair.default_recursion_depth_limit",
                               error) ||
      !requirePositiveUnsigned(opts.localRepair.hotspotLimit,
                               "local_repair.hotspot_limit", error) ||
      !requirePositiveUnsigned(opts.localRepair.repairRadiusStep,
                               "local_repair.repair_radius_step", error) ||
      !requirePositiveUnsigned(opts.localRepair.repairRadiusBias,
                               "local_repair.repair_radius_bias", error) ||
      !requirePositiveUnsigned(opts.localRepair.relocationCandidateLimit,
                               "local_repair.relocation_candidate_limit",
                               error) ||
      !requirePositiveUnsigned(opts.localRepair.swapCandidateLimit,
                               "local_repair.swap_candidate_limit",
                               error) ||
      !requirePositiveUnsigned(opts.localRepair.earlyCPSatNeighborhoodLimit,
                               "local_repair.early_cpsat_neighborhood_limit",
                               error) ||
      !requirePositiveUnsigned(opts.localRepair.earlyCPSatMoveRadius,
                               "local_repair.early_cpsat_move_radius", error) ||
      !requirePositiveDouble(opts.localRepair.failedEdgeDeltaScoreWeight,
                             "local_repair.failed_edge_delta_score_weight",
                             error) ||
      !requirePositiveDouble(opts.localRepair.candidateScoreWeight,
                             "local_repair.candidate_score_weight", error) ||
      !requirePositiveDouble(opts.localRepair.placementScoreWeight,
                             "local_repair.placement_score_weight", error) ||
      !requirePositiveDouble(opts.localRepair.hotspotDistanceScoreWeight,
                             "local_repair.hotspot_distance_score_weight",
                             error) ||
      !requirePositiveDouble(
          opts.localRepair.hotspotSourceDistanceScoreWeight,
          "local_repair.hotspot_source_distance_score_weight", error) ||
      !requirePositiveDouble(opts.localRepair.cpSatSmallFailedMinTime,
                             "local_repair.cpsat_small_failed_min_time",
                             error) ||
      !requirePositiveDouble(opts.localRepair.cpSatMediumFailedMinTime,
                             "local_repair.cpsat_medium_failed_min_time",
                             error) ||
      !requirePositiveUnsigned(
          opts.localRepair.repairNegotiatedRoutingPassCap,
          "local_repair.repair_negotiated_routing_pass_cap", error) ||
      !requirePositiveUnsigned(opts.localRepair.freePathMissingPenalty,
                               "local_repair.free_path_missing_penalty",
                               error) ||
      !requirePositiveUnsigned(opts.localRepair.exactDomainSearchSpaceCap,
                               "local_repair.exact_domain_search_space_cap",
                               error) ||
      !requirePositiveUnsigned(
          opts.localRepair.memoryExactDomainSearchSpaceCap,
          "local_repair.memory_exact_domain_search_space_cap", error) ||
      !requirePositiveDouble(opts.localRepair.focusNeighborhoodWeightScale,
                             "local_repair.focus_neighborhood_weight_scale",
                             error) ||
      !requirePositiveUnsigned(opts.localRepair.exactNeighborhoodRadius,
                               "local_repair.exact_neighborhood_radius",
                               error) ||
      !requirePositiveUnsigned(
          opts.localRepair.exactNeighborhoodSearchSpaceTightCap,
          "local_repair.exact_neighborhood_search_space_tight_cap", error) ||
      !requirePositiveUnsigned(
          opts.localRepair.exactNeighborhoodSearchSpaceDefaultCap,
          "local_repair.exact_neighborhood_search_space_default_cap", error) ||
      !requirePositiveUnsigned(opts.localRepair.memoryFocusNodeLimit,
                               "local_repair.memory_focus_node_limit",
                               error) ||
      !requirePositiveDouble(opts.localRepair.focusedTargetMinTime,
                             "local_repair.focused_target_min_time", error) ||
      !requirePositiveDouble(opts.localRepair.cycleExactMinTime,
                             "local_repair.cycle_exact_min_time", error) ||
      !requirePositiveDouble(opts.localRepair.focusedLocalHistoryBump,
                             "local_repair.focused_local_history_bump",
                             error) ||
      !requirePositiveUnsigned(opts.localRepair.jointRipupMaxEdges,
                               "local_repair.joint_ripup_max_edges", error) ||
      !requirePositiveUnsigned(opts.localRepair.singleRipupMaxEdges,
                               "local_repair.single_ripup_max_edges", error)) {
    return false;
  }

  if (opts.localRepair.memoryResponseClusterMax <
      opts.localRepair.memoryResponseClusterMin) {
    error = "local_repair.memory_response_cluster_max must be >= min";
    return false;
  }
  if (opts.localRepair.singleRipupMaxEdges <
      opts.localRepair.singleRipupMinEdges) {
    error = "local_repair.single_ripup_max_edges must be >= min";
    return false;
  }
  return true;
}

} // namespace fcc
