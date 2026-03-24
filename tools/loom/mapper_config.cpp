#include "mapper_config.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/YAMLTraits.h"

namespace loom {

namespace {

struct MapperBaseConfigDocument {
  unsigned version = 1;
  MapperOptions mapper;
};

} // namespace

} // namespace loom

namespace llvm {
namespace yaml {

template <> struct MappingTraits<loom::MapperRefinementOptions> {
  static void mapping(IO &io, loom::MapperRefinementOptions &opts) {
    io.mapOptional("initial_temperature", opts.initialTemperature);
    io.mapOptional("cooling_rate", opts.coolingRate);
    io.mapOptional("adaptive_cooling_enabled", opts.adaptiveCoolingEnabled);
    io.mapOptional("route_aware_neighborhood_enabled",
                   opts.routeAwareNeighborhoodEnabled);
    io.mapOptional("route_aware_checkpoint_enabled",
                   opts.routeAwareCheckpointEnabled);
    io.mapOptional("adaptive_window", opts.adaptiveWindow);
    io.mapOptional("route_aware_neighborhood_edge_cap",
                   opts.routeAwareNeighborhoodEdgeCap);
    io.mapOptional("route_aware_checkpoint_accepted_move_batch",
                   opts.routeAwareCheckpointAcceptedMoveBatch);
    io.mapOptional("target_acceptance_low", opts.targetAcceptanceLow);
    io.mapOptional("target_acceptance_high", opts.targetAcceptanceHigh);
    io.mapOptional("cold_acceptance_reheat_multiplier",
                   opts.coldAcceptanceReheatMultiplier);
    io.mapOptional("hot_acceptance_cooling_multiplier",
                   opts.hotAcceptanceCoolingMultiplier);
    io.mapOptional("plateau_window", opts.plateauWindow);
    io.mapOptional("plateau_reheat_multiplier", opts.plateauReheatMultiplier);
    io.mapOptional("max_temperature_scale", opts.maxTemperatureScale);
    io.mapOptional("min_temperature", opts.minTemperature);
    io.mapOptional("iterations_per_placed_node", opts.iterationsPerPlacedNode);
    io.mapOptional("iteration_cap", opts.iterationCap);
    io.mapOptional("budget_fraction", opts.budgetFraction);
    io.mapOptional("relocate_top_candidate_limit",
                   opts.relocateTopCandidateLimit);
    io.mapOptional("route_aware_sa_budget_fraction",
                   opts.routeAwareSABudgetFraction);
    io.mapOptional("warmup_budget_fraction", opts.warmupBudgetFraction);
    io.mapOptional("initial_routing_budget_fraction",
                   opts.initialRoutingBudgetFraction);
    io.mapOptional("route_aware_sa_neighborhood_edge_cap",
                   opts.routeAwareSANeighborhoodEdgeCap);
    io.mapOptional("route_aware_sa_exact_repair_micro_budget_ms",
                   opts.routeAwareSAExactRepairMicroBudgetMs);
    io.mapOptional("route_aware_sa_checkpoint_move_batch",
                   opts.routeAwareSACheckpointMoveBatch);
    io.mapOptional("route_aware_sa_initial_temperature",
                   opts.routeAwareSAInitialTemperature);
    io.mapOptional("route_aware_sa_cooling_rate",
                   opts.routeAwareSACoolingRate);
    io.mapOptional("route_aware_sa_min_temperature",
                   opts.routeAwareSAMinTemperature);
  }
};

template <> struct MappingTraits<loom::MapperLaneOptions> {
  static void mapping(IO &io, loom::MapperLaneOptions &opts) {
    io.mapOptional("auto_serial_node_threshold", opts.autoSerialNodeThreshold);
    io.mapOptional("auto_lane_cap", opts.autoLaneCap);
    io.mapOptional("routing_beam_width", opts.routingBeamWidth);
    io.mapOptional("final_polish_reserve_fraction",
                   opts.finalPolishReserveFraction);
    io.mapOptional("lane_seed_stride", opts.laneSeedStride);
    io.mapOptional("restart_seed_stride", opts.restartSeedStride);
    io.mapOptional("restart_move_radius_step", opts.restartMoveRadiusStep);
    io.mapOptional("restart_ripup_bonus_cap", opts.restartRipupBonusCap);
    io.mapOptional("global_cpsat_min_time_single_lane",
                   opts.globalCPSatMinTimeSingleLane);
    io.mapOptional("global_cpsat_min_time_multi_lane",
                   opts.globalCPSatMinTimeMultiLane);
    io.mapOptional("boundary_cpsat_min_time_single_lane",
                   opts.boundaryCPSatMinTimeSingleLane);
    io.mapOptional("boundary_cpsat_min_time_multi_lane",
                   opts.boundaryCPSatMinTimeMultiLane);
    io.mapOptional("boundary_neighborhood_cap", opts.boundaryNeighborhoodCap);
    io.mapOptional("unrouted_diagnostic_limit", opts.unroutedDiagnosticLimit);
  }
};

template <> struct MappingTraits<loom::MapperRoutingPriorityOptions> {
  static void mapping(IO &io, loom::MapperRoutingPriorityOptions &opts) {
    io.mapOptional("invalid_edge", opts.invalidEdge);
    io.mapOptional("memory_sink", opts.memorySink);
    io.mapOptional("module_output", opts.moduleOutput);
    io.mapOptional("load_store_source", opts.loadStoreSource);
    io.mapOptional("memory_source", opts.memorySource);
    io.mapOptional("load_store_incident", opts.loadStoreIncident);
    io.mapOptional("fallback", opts.fallback);
  }
};

template <> struct MappingTraits<loom::MapperRoutingStrategyOptions> {
  static void mapping(IO &io, loom::MapperRoutingStrategyOptions &opts) {
    io.mapOptional("priority", opts.priority);
    io.mapOptional("tight_ripup_failed_edge_threshold",
                   opts.tightRipupFailedEdgeThreshold);
    io.mapOptional("sibling_expansion_failed_edge_threshold",
                   opts.siblingExpansionFailedEdgeThreshold);
    io.mapOptional("tight_ripup_min_edges", opts.tightRipupMinEdges);
    io.mapOptional("tight_ripup_target_cap", opts.tightRipupTargetCap);
    io.mapOptional("failed_edge_diagnostic_limit",
                   opts.failedEdgeDiagnosticLimit);
    io.mapOptional("module_output_group_max_edges",
                   opts.moduleOutputGroupMaxEdges);
    io.mapOptional("module_output_first_hop_limit",
                   opts.moduleOutputFirstHopLimit);
    io.mapOptional("module_output_branch_limit",
                   opts.moduleOutputBranchLimit);
    io.mapOptional("source_fanout_first_hop_limit",
                   opts.sourceFanoutFirstHopLimit);
    io.mapOptional("ripup_history_bump", opts.ripupHistoryBump);
  }
};

template <> struct MappingTraits<loom::MapperCongestionOptions> {
  static void mapping(IO &io, loom::MapperCongestionOptions &opts) {
    io.mapOptional("saturation_penalty", opts.saturationPenalty);
    io.mapOptional("history_increment_cap", opts.historyIncrementCap);
    io.mapOptional("history_decay", opts.historyDecay);
    io.mapOptional("routing_output_history_bump",
                   opts.routingOutputHistoryBump);
    io.mapOptional("routing_output_history_decay",
                   opts.routingOutputHistoryDecay);
    io.mapOptional("early_termination_window", opts.earlyTerminationWindow);
  }
};

template <> struct MappingTraits<loom::MapperTimingOptions> {
  static void mapping(IO &io, loom::MapperTimingOptions &opts) {
    io.mapOptional("recurrence_edge_weight_multiplier",
                   opts.recurrenceEdgeWeightMultiplier);
    io.mapOptional("recurrence_node_latency_weight",
                   opts.recurrenceNodeLatencyWeight);
    io.mapOptional("recurrence_node_interval_weight",
                   opts.recurrenceNodeIntervalWeight);
    io.mapOptional("combinational_node_delay", opts.combinationalNodeDelay);
    io.mapOptional("routing_hop_delay", opts.routingHopDelay);
  }
};

template <> struct MappingTraits<loom::MapperBufferizationOptions> {
  static void mapping(IO &io, loom::MapperBufferizationOptions &opts) {
    io.mapOptional("enabled", opts.enabled);
    io.mapOptional("max_iterations", opts.maxIterations);
    io.mapOptional("outer_joint_iterations", opts.outerJointIterations);
    io.mapOptional("min_throughput_improvement",
                   opts.minThroughputImprovement);
    io.mapOptional("clock_tie_break_improvement",
                   opts.clockTieBreakImprovement);
  }
};

template <> struct MappingTraits<loom::MapperTechFeedbackOptions> {
  static void mapping(IO &io, loom::MapperTechFeedbackOptions &opts) {
    io.mapOptional("enabled", opts.enabled);
    io.mapOptional("max_retries", opts.maxRetries);
    io.mapOptional("max_targets_per_retry", opts.maxTargetsPerRetry);
  }
};

template <> struct MappingTraits<loom::MapperRelaxedRoutingOptions> {
  static void mapping(IO &io, loom::MapperRelaxedRoutingOptions &opts) {
    io.mapOptional("enabled", opts.enabled);
    io.mapOptional("legalization_passes", opts.legalizationPasses);
    io.mapOptional("base_overuse_penalty", opts.baseOverusePenalty);
    io.mapOptional("repeated_overuse_scale", opts.repeatedOveruseScale);
    io.mapOptional("reject_checkpoint_overuse_cap",
                   opts.rejectCheckpointOveruseCap);
  }
};

template <> struct MappingTraits<loom::MapperCPSatTuningOptions> {
  static void mapping(IO &io, loom::MapperCPSatTuningOptions &opts) {
    io.mapOptional("placement_cost_scale", opts.placementCostScale);
    io.mapOptional("worker_fallback_concurrency",
                   opts.workerFallbackConcurrency);
    io.mapOptional("worker_cap", opts.workerCap);
    io.mapOptional("global_candidate_limit", opts.globalCandidateLimit);
    io.mapOptional("tight_endgame_failed_edge_threshold",
                   opts.tightEndgameFailedEdgeThreshold);
    io.mapOptional("very_tight_endgame_failed_edge_threshold",
                   opts.veryTightEndgameFailedEdgeThreshold);
    io.mapOptional("neighborhood_expansion_edge_multiplier",
                   opts.neighborhoodExpansionEdgeMultiplier);
    io.mapOptional("neighborhood_expansion_base",
                   opts.neighborhoodExpansionBase);
    io.mapOptional("neighborhood_expansion_cap", opts.neighborhoodExpansionCap);
    io.mapOptional("very_tight_focus_weight_floor",
                   opts.veryTightFocusWeightFloor);
    io.mapOptional("very_tight_focus_weight_scale",
                   opts.veryTightFocusWeightScale);
    io.mapOptional("tight_focus_weight_floor", opts.tightFocusWeightFloor);
    io.mapOptional("tight_focus_weight_scale", opts.tightFocusWeightScale);
    io.mapOptional("tight_move_radius_floor", opts.tightMoveRadiusFloor);
    io.mapOptional("very_tight_move_radius_floor",
                   opts.veryTightMoveRadiusFloor);
    io.mapOptional("move_radius_expansion", opts.moveRadiusExpansion);
    io.mapOptional("very_tight_move_radius_expansion",
                   opts.veryTightMoveRadiusExpansion);
    io.mapOptional("very_tight_candidate_limit",
                   opts.veryTightCandidateLimit);
    io.mapOptional("tight_candidate_limit", opts.tightCandidateLimit);
    io.mapOptional("default_candidate_limit", opts.defaultCandidateLimit);
  }
};

template <> struct MappingTraits<loom::MapperLocalRepairExactOptions> {
  static void mapping(IO &io, loom::MapperLocalRepairExactOptions &opts) {
    io.mapOptional("priority_first_failed_edge_threshold",
                   opts.priorityFirstFailedEdgeThreshold);
    io.mapOptional("neighborhood_pass_min", opts.neighborhoodPassMin);
    io.mapOptional("neighborhood_pass_cap", opts.neighborhoodPassCap);
    io.mapOptional("tight_failed_edge_threshold", opts.tightFailedEdgeThreshold);
    io.mapOptional("tight_repair_edge_threshold", opts.tightRepairEdgeThreshold);
    io.mapOptional("micro_failed_edge_threshold", opts.microFailedEdgeThreshold);
    io.mapOptional("micro_repair_edge_threshold", opts.microRepairEdgeThreshold);
    io.mapOptional("micro_deadline_ms", opts.microDeadlineMs);
    io.mapOptional("tight_deadline_ms", opts.tightDeadlineMs);
    io.mapOptional("medium_repair_edge_threshold",
                   opts.mediumRepairEdgeThreshold);
    io.mapOptional("medium_deadline_ms", opts.mediumDeadlineMs);
    io.mapOptional("default_deadline_ms", opts.defaultDeadlineMs);
    io.mapOptional("deadline_scale", opts.deadlineScale);
    io.mapOptional("micro_deadline_scale", opts.microDeadlineScale);
    io.mapOptional("micro_first_hop_limit", opts.microFirstHopLimit);
    io.mapOptional("tight_first_hop_limit", opts.tightFirstHopLimit);
    io.mapOptional("medium_first_hop_limit", opts.mediumFirstHopLimit);
    io.mapOptional("small_failed_first_hop_limit",
                   opts.smallFailedFirstHopLimit);
    io.mapOptional("default_first_hop_limit", opts.defaultFirstHopLimit);
    io.mapOptional("micro_candidate_path_limit",
                   opts.microCandidatePathLimit);
    io.mapOptional("tight_candidate_path_limit",
                   opts.tightCandidatePathLimit);
    io.mapOptional("medium_candidate_path_limit",
                   opts.mediumCandidatePathLimit);
    io.mapOptional("small_failed_candidate_path_limit",
                   opts.smallFailedCandidatePathLimit);
    io.mapOptional("default_candidate_path_limit",
                   opts.defaultCandidatePathLimit);
    io.mapOptional("local_history_bump", opts.localHistoryBump);
  }
};

template <> struct MappingTraits<loom::MapperLocalRepairOptions> {
  static void mapping(IO &io, loom::MapperLocalRepairOptions &opts) {
    io.mapOptional("enabled", opts.enabled);
    io.mapOptional("exact", opts.exact);
    io.mapOptional("micro_recursion_depth_limit",
                   opts.microRecursionDepthLimit);
    io.mapOptional("default_recursion_depth_limit",
                   opts.defaultRecursionDepthLimit);
    io.mapOptional("original_failed_escalation_threshold",
                   opts.originalFailedEscalationThreshold);
    io.mapOptional("small_failed_edge_threshold",
                   opts.smallFailedEdgeThreshold);
    io.mapOptional("hotspot_limit", opts.hotspotLimit);
    io.mapOptional("hotspot_adjacency_radius", opts.hotspotAdjacencyRadius);
    io.mapOptional("repair_radius_step", opts.repairRadiusStep);
    io.mapOptional("repair_radius_bias", opts.repairRadiusBias);
    io.mapOptional("relocation_candidate_limit",
                   opts.relocationCandidateLimit);
    io.mapOptional("swap_candidate_limit", opts.swapCandidateLimit);
    io.mapOptional("early_cpsat_recursion_depth_threshold",
                   opts.earlyCPSatRecursionDepthThreshold);
    io.mapOptional("early_cpsat_failed_edge_threshold",
                   opts.earlyCPSatFailedEdgeThreshold);
    io.mapOptional("early_cpsat_min_time", opts.earlyCPSatMinTime);
    io.mapOptional("early_cpsat_neighborhood_limit",
                   opts.earlyCPSatNeighborhoodLimit);
    io.mapOptional("early_cpsat_move_radius", opts.earlyCPSatMoveRadius);
    io.mapOptional("failed_edge_delta_score_weight",
                   opts.failedEdgeDeltaScoreWeight);
    io.mapOptional("candidate_score_weight", opts.candidateScoreWeight);
    io.mapOptional("placement_score_weight", opts.placementScoreWeight);
    io.mapOptional("hotspot_distance_score_weight",
                   opts.hotspotDistanceScoreWeight);
    io.mapOptional("hotspot_source_distance_score_weight",
                   opts.hotspotSourceDistanceScoreWeight);
    io.mapOptional("cpsat_small_failed_threshold",
                   opts.cpSatSmallFailedThreshold);
    io.mapOptional("cpsat_medium_failed_threshold",
                   opts.cpSatMediumFailedThreshold);
    io.mapOptional("cpsat_small_failed_min_time",
                   opts.cpSatSmallFailedMinTime);
    io.mapOptional("cpsat_medium_failed_min_time",
                   opts.cpSatMediumFailedMinTime);
    io.mapOptional("cpsat_small_failed_node_limit",
                   opts.cpSatSmallFailedNodeLimit);
    io.mapOptional("cpsat_medium_failed_node_limit",
                   opts.cpSatMediumFailedNodeLimit);
    io.mapOptional("repair_negotiated_routing_pass_cap",
                   opts.repairNegotiatedRoutingPassCap);
    io.mapOptional("free_path_missing_penalty",
                   opts.freePathMissingPenalty);
    io.mapOptional("memory_response_cluster_min",
                   opts.memoryResponseClusterMin);
    io.mapOptional("memory_response_cluster_max",
                   opts.memoryResponseClusterMax);
    io.mapOptional("exact_domain_search_space_cap",
                   opts.exactDomainSearchSpaceCap);
    io.mapOptional("memory_exact_domain_search_space_cap",
                   opts.memoryExactDomainSearchSpaceCap);
    io.mapOptional("focus_neighborhood_weight_scale",
                   opts.focusNeighborhoodWeightScale);
    io.mapOptional("exact_neighborhood_radius",
                   opts.exactNeighborhoodRadius);
    io.mapOptional("exact_neighborhood_search_space_tight_cap",
                   opts.exactNeighborhoodSearchSpaceTightCap);
    io.mapOptional("exact_neighborhood_search_space_default_cap",
                   opts.exactNeighborhoodSearchSpaceDefaultCap);
    io.mapOptional("large_neighborhood_failed_edge_threshold",
                   opts.largeNeighborhoodFailedEdgeThreshold);
    io.mapOptional("cpsat_fallback_failed_edge_threshold",
                   opts.cpSatFallbackFailedEdgeThreshold);
    io.mapOptional("cpsat_escalation_failed_edge_threshold",
                   opts.cpSatEscalationFailedEdgeThreshold);
    io.mapOptional("target_focused_failed_edge_threshold",
                   opts.targetFocusedFailedEdgeThreshold);
    io.mapOptional("focused_target_edge_threshold",
                   opts.focusedTargetEdgeThreshold);
    io.mapOptional("focused_blocker_edge_threshold",
                   opts.focusedBlockerEdgeThreshold);
    io.mapOptional("memory_focus_node_limit", opts.memoryFocusNodeLimit);
    io.mapOptional("focused_target_min_time", opts.focusedTargetMinTime);
    io.mapOptional("residual_repair_failed_edge_threshold",
                   opts.residualRepairFailedEdgeThreshold);
    io.mapOptional("residual_joint_repair_failed_edge_threshold",
                   opts.residualJointRepairFailedEdgeThreshold);
    io.mapOptional("cycle_edge_cluster_min", opts.cycleEdgeClusterMin);
    io.mapOptional("cycle_edge_cluster_max", opts.cycleEdgeClusterMax);
    io.mapOptional("cycle_exact_min_time", opts.cycleExactMinTime);
    io.mapOptional("focused_local_history_bump",
                   opts.focusedLocalHistoryBump);
    io.mapOptional("joint_ripup_max_edges", opts.jointRipupMaxEdges);
    io.mapOptional("single_ripup_max_edges", opts.singleRipupMaxEdges);
    io.mapOptional("single_ripup_min_edges", opts.singleRipupMinEdges);
  }
};

template <> struct MappingTraits<loom::MapperOptions> {
  static void mapping(IO &io, loom::MapperOptions &opts) {
    io.mapOptional("budget_seconds", opts.budgetSeconds);
    io.mapOptional("seed", opts.seed);
    io.mapOptional("profile", opts.profile);
    io.mapOptional("lanes", opts.lanes);
    io.mapOptional("snapshot_interval_seconds", opts.snapshotIntervalSeconds);
    io.mapOptional("snapshot_interval_rounds", opts.snapshotIntervalRounds);
    io.mapOptional("interleaved_rounds", opts.interleavedRounds);
    io.mapOptional("selective_ripup_passes", opts.selectiveRipupPasses);
    io.mapOptional("placement_move_radius", opts.placementMoveRadius);
    io.mapOptional("cpsat_global_node_limit", opts.cpSatGlobalNodeLimit);
    io.mapOptional("cpsat_neighborhood_node_limit",
                   opts.cpSatNeighborhoodNodeLimit);
    io.mapOptional("cpsat_time_limit_seconds", opts.cpSatTimeLimitSeconds);
    io.mapOptional("enable_cpsat", opts.enableCPSat);
    io.mapOptional("verbose", opts.verbose);
    io.mapOptional("routing_heuristic_weight", opts.routingHeuristicWeight);
    io.mapOptional("negotiated_routing_passes", opts.negotiatedRoutingPasses);
    io.mapOptional("congestion_history_factor", opts.congestionHistoryFactor);
    io.mapOptional("congestion_history_scale", opts.congestionHistoryScale);
    io.mapOptional("congestion_present_factor", opts.congestionPresentFactor);
    io.mapOptional("congestion_placement_weight",
                   opts.congestionPlacementWeight);
    io.mapOptional("memory_sharing_penalty", opts.memorySharingPenalty);
    io.mapOptional("refinement", opts.refinement);
    io.mapOptional("lane", opts.lane);
    io.mapOptional("routing", opts.routing);
    io.mapOptional("congestion", opts.congestion);
    io.mapOptional("timing", opts.timing);
    io.mapOptional("bufferization", opts.bufferization);
    io.mapOptional("tech_feedback", opts.techFeedback);
    io.mapOptional("relaxed_routing", opts.relaxedRouting);
    io.mapOptional("cpsat_tuning", opts.cpSatTuning);
    io.mapOptional("local_repair", opts.localRepair);
    io.mapOptional("enable_route_aware_sa_main_loop",
                   opts.enableRouteAwareSAMainLoop);
  }
};

template <> struct MappingTraits<loom::MapperBaseConfigDocument> {
  static void mapping(IO &io, loom::MapperBaseConfigDocument &doc) {
    io.mapOptional("version", doc.version, 1u);
    io.mapRequired("mapper", doc.mapper);
  }
};

} // namespace yaml
} // namespace llvm

namespace loom {

std::string getDefaultMapperBaseConfigPath() {
#ifdef LOOM_SOURCE_DIR
  llvm::SmallString<256> path(LOOM_SOURCE_DIR);
  llvm::sys::path::append(path, "configs", "mapper", "default.yaml");
  return std::string(path.str());
#else
  return {};
#endif
}

bool loadMapperBaseConfig(const std::string &path, MapperOptions &opts,
                         std::string &error) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrErr) {
    error = "cannot open mapper base config '" + path + "'";
    return false;
  }

  MapperBaseConfigDocument doc;
  doc.mapper = opts;
  llvm::yaml::Input input((*bufferOrErr)->getBuffer());
  input >> doc;
  if (std::error_code ec = input.error()) {
    error = "failed to parse mapper base config '" + path +
            "': " + ec.message();
    return false;
  }
  if (doc.version != 1) {
    error = "unsupported mapper base config version '" +
            std::to_string(doc.version) + "'";
    return false;
  }
  if (!validateMapperOptions(doc.mapper, error)) {
    error = "invalid mapper base config '" + path + "': " + error;
    return false;
  }
  opts = std::move(doc.mapper);
  return true;
}

} // namespace loom
