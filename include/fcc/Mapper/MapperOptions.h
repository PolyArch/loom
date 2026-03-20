#ifndef FCC_MAPPER_MAPPER_OPTIONS_H
#define FCC_MAPPER_MAPPER_OPTIONS_H

#include <string>

namespace fcc {

struct MapperRefinementOptions {
  double initialTemperature = 100.0;
  double coolingRate = 0.995;
  unsigned iterationsPerPlacedNode = 1000;
  unsigned iterationCap = 50000;
  double budgetFraction = 0.4;
  unsigned relocateTopCandidateLimit = 8;
};

struct MapperLaneOptions {
  unsigned autoSerialNodeThreshold = 16;
  unsigned autoLaneCap = 4;
  double finalPolishReserveFraction = 0.15;
  unsigned laneSeedStride = 7919;
  unsigned restartSeedStride = 104729;
  unsigned restartMoveRadiusStep = 2;
  unsigned restartRipupBonusCap = 2;
  double globalCPSatMinTimeSingleLane = 4.0;
  double globalCPSatMinTimeMultiLane = 2.0;
  double boundaryCPSatMinTimeSingleLane = 3.0;
  double boundaryCPSatMinTimeMultiLane = 1.5;
  unsigned boundaryNeighborhoodCap = 8;
  unsigned unroutedDiagnosticLimit = 8;
};

struct MapperRoutingPriorityOptions {
  unsigned invalidEdge = 100;
  unsigned memorySink = 0;
  unsigned moduleOutput = 1;
  unsigned loadStoreSource = 2;
  unsigned memorySource = 3;
  unsigned loadStoreIncident = 4;
  unsigned fallback = 5;
};

struct MapperRoutingStrategyOptions {
  MapperRoutingPriorityOptions priority;
  unsigned tightRipupFailedEdgeThreshold = 4;
  unsigned siblingExpansionFailedEdgeThreshold = 6;
  unsigned tightRipupMinEdges = 8;
  unsigned tightRipupTargetCap = 18;
  unsigned failedEdgeDiagnosticLimit = 6;
  unsigned moduleOutputGroupMaxEdges = 8;
  unsigned moduleOutputFirstHopLimit = 2;
  unsigned moduleOutputBranchLimit = 2;
  unsigned sourceFanoutFirstHopLimit = 4;
  double ripupHistoryBump = 0.35;
};

struct MapperCongestionOptions {
  double saturationPenalty = 0.5;
  double historyIncrementCap = 8.0;
  double historyDecay = 0.95;
  double routingOutputHistoryBump = 0.2;
  double routingOutputHistoryDecay = 0.95;
  unsigned earlyTerminationWindow = 3;
};

struct MapperCPSatTuningOptions {
  double placementCostScale = 100.0;
  unsigned workerFallbackConcurrency = 4;
  unsigned workerCap = 8;
  unsigned globalCandidateLimit = 8;
  unsigned tightEndgameFailedEdgeThreshold = 6;
  unsigned veryTightEndgameFailedEdgeThreshold = 2;
  unsigned neighborhoodExpansionEdgeMultiplier = 2;
  unsigned neighborhoodExpansionBase = 2;
  unsigned neighborhoodExpansionCap = 12;
  double veryTightFocusWeightFloor = 80.0;
  double veryTightFocusWeightScale = 28.0;
  double tightFocusWeightFloor = 40.0;
  double tightFocusWeightScale = 18.0;
  unsigned tightMoveRadiusFloor = 3;
  unsigned veryTightMoveRadiusFloor = 5;
  unsigned moveRadiusExpansion = 1;
  unsigned veryTightMoveRadiusExpansion = 2;
  unsigned veryTightCandidateLimit = 12;
  unsigned tightCandidateLimit = 5;
  unsigned defaultCandidateLimit = 8;
};

struct MapperLocalRepairExactOptions {
  unsigned priorityFirstFailedEdgeThreshold = 2;
  unsigned neighborhoodPassMin = 2;
  unsigned neighborhoodPassCap = 6;
  unsigned tightFailedEdgeThreshold = 3;
  unsigned tightRepairEdgeThreshold = 6;
  unsigned microFailedEdgeThreshold = 2;
  unsigned microRepairEdgeThreshold = 6;
  double microDeadlineMs = 20000.0;
  double tightDeadlineMs = 8000.0;
  unsigned mediumRepairEdgeThreshold = 10;
  double mediumDeadlineMs = 4500.0;
  double defaultDeadlineMs = 3000.0;
  double deadlineScale = 4000.0;
  double microDeadlineScale = 8000.0;
  unsigned microFirstHopLimit = 20;
  unsigned tightFirstHopLimit = 12;
  unsigned mediumFirstHopLimit = 8;
  unsigned smallFailedFirstHopLimit = 6;
  unsigned defaultFirstHopLimit = 5;
  unsigned microCandidatePathLimit = 32;
  unsigned tightCandidatePathLimit = 16;
  unsigned mediumCandidatePathLimit = 8;
  unsigned smallFailedCandidatePathLimit = 6;
  unsigned defaultCandidatePathLimit = 5;
  double localHistoryBump = 1.5;
};

struct MapperLocalRepairOptions {
  MapperLocalRepairExactOptions exact;
  unsigned microRecursionDepthLimit = 4;
  unsigned defaultRecursionDepthLimit = 2;
  unsigned originalFailedEscalationThreshold = 12;
  unsigned smallFailedEdgeThreshold = 3;
  unsigned hotspotLimit = 12;
  unsigned hotspotAdjacencyRadius = 1;
  unsigned repairRadiusStep = 2;
  unsigned repairRadiusBias = 1;
  unsigned relocationCandidateLimit = 10;
  unsigned swapCandidateLimit = 8;
  unsigned earlyCPSatRecursionDepthThreshold = 2;
  unsigned earlyCPSatFailedEdgeThreshold = 2;
  double earlyCPSatMinTime = 12.0;
  unsigned earlyCPSatNeighborhoodLimit = 12;
  unsigned earlyCPSatMoveRadius = 5;
  double failedEdgeDeltaScoreWeight = 8.0;
  double candidateScoreWeight = 0.25;
  double placementScoreWeight = 8.0;
  double hotspotDistanceScoreWeight = 0.25;
  double hotspotSourceDistanceScoreWeight = 0.25;
  unsigned cpSatSmallFailedThreshold = 4;
  unsigned cpSatMediumFailedThreshold = 6;
  double cpSatSmallFailedMinTime = 8.0;
  double cpSatMediumFailedMinTime = 8.0;
  unsigned cpSatSmallFailedNodeLimit = 8;
  unsigned cpSatMediumFailedNodeLimit = 8;
  unsigned repairNegotiatedRoutingPassCap = 4;
  unsigned freePathMissingPenalty = 1000;
  unsigned memoryResponseClusterMin = 3;
  unsigned memoryResponseClusterMax = 6;
  unsigned exactDomainSearchSpaceCap = 1296;
  unsigned memoryExactDomainSearchSpaceCap = 1024;
  double focusNeighborhoodWeightScale = 0.35;
  unsigned exactNeighborhoodRadius = 4;
  unsigned exactNeighborhoodSearchSpaceTightCap = 16384;
  unsigned exactNeighborhoodSearchSpaceDefaultCap = 8192;
  unsigned largeNeighborhoodFailedEdgeThreshold = 4;
  unsigned cpSatFallbackFailedEdgeThreshold = 8;
  unsigned cpSatEscalationFailedEdgeThreshold = 6;
  unsigned targetFocusedFailedEdgeThreshold = 3;
  unsigned focusedTargetEdgeThreshold = 3;
  unsigned focusedBlockerEdgeThreshold = 12;
  unsigned memoryFocusNodeLimit = 4;
  double focusedTargetMinTime = 2.0;
  unsigned residualRepairFailedEdgeThreshold = 6;
  unsigned residualJointRepairFailedEdgeThreshold = 2;
  unsigned cycleEdgeClusterMin = 3;
  unsigned cycleEdgeClusterMax = 4;
  double cycleExactMinTime = 1.25;
  double focusedLocalHistoryBump = 2.0;
  unsigned jointRipupMaxEdges = 20;
  unsigned singleRipupMaxEdges = 18;
  unsigned singleRipupMinEdges = 1;
};

struct MapperOptions {
  double budgetSeconds = 60.0;
  int seed = 0;
  std::string profile = "balanced";
  unsigned lanes = 0;
  double snapshotIntervalSeconds = -1.0;
  int snapshotIntervalRounds = -1;
  unsigned interleavedRounds = 4;
  unsigned selectiveRipupPasses = 3;
  unsigned placementMoveRadius = 3;
  unsigned cpSatGlobalNodeLimit = 24;
  unsigned cpSatNeighborhoodNodeLimit = 8;
  double cpSatTimeLimitSeconds = 0.75;
  bool enableCPSat = true;
  bool verbose = true;
  double routingHeuristicWeight = 1.5;
  unsigned negotiatedRoutingPasses = 12;
  double congestionHistoryFactor = 1.0;
  double congestionHistoryScale = 1.5;
  double congestionPresentFactor = 1.0;
  double congestionPlacementWeight = 0.3;
  double memorySharingPenalty = 8.0;

  MapperRefinementOptions refinement;
  MapperLaneOptions lane;
  MapperRoutingStrategyOptions routing;
  MapperCongestionOptions congestion;
  MapperCPSatTuningOptions cpSatTuning;
  MapperLocalRepairOptions localRepair;
};

bool validateMapperOptions(const MapperOptions &opts, std::string &error);

} // namespace fcc

#endif
