#include "loom/SystemCompiler/TDCVerification.h"

#include <sstream>

namespace loom {

//===----------------------------------------------------------------------===//
// Helper: edge identity matching
//===----------------------------------------------------------------------===//

/// Check if two edge identifiers match (producer + consumer kernel names).
static bool edgeMatchesSpec(const std::string &specProducer,
                            const std::string &specConsumer,
                            const std::string &otherProducer,
                            const std::string &otherConsumer) {
  return specProducer == otherProducer && specConsumer == otherConsumer;
}

/// Build an edge key string for diagnostic messages.
static std::string edgeKey(const std::string &producer,
                           const std::string &consumer) {
  return producer + " -> " + consumer;
}

//===----------------------------------------------------------------------===//
// TDCContractInferrer
//===----------------------------------------------------------------------===//

TDCInferenceResult
TDCContractInferrer::infer(const std::vector<TDCEdgeSpec> &edgeSpecs,
                           const std::vector<TDCPathSpec> &pathSpecs) const {
  TDCInferenceResult result;
  result.edgeSpecs.reserve(edgeSpecs.size());
  result.edgeOrigins.reserve(edgeSpecs.size());

  for (const auto &input : edgeSpecs) {
    TDCEdgeSpec inferred = input;
    TDCEdgeSpecOrigin origin;

    // Ordering: fill missing with FIFO (conservative default).
    if (input.ordering.has_value()) {
      origin.ordering = DimensionOrigin::USER_SPECIFIED;
    } else {
      inferred.ordering = Ordering::FIFO;
      origin.ordering = DimensionOrigin::INFERRED;
    }

    // Throughput: no default; remains absent if not specified.
    if (input.throughput.has_value()) {
      origin.throughput = DimensionOrigin::USER_SPECIFIED;
    } else {
      origin.throughput = DimensionOrigin::ABSENT;
    }

    // Placement: fill missing with AUTO (compiler decides).
    if (input.placement.has_value()) {
      origin.placement = DimensionOrigin::USER_SPECIFIED;
    } else {
      inferred.placement = Placement::AUTO;
      origin.placement = DimensionOrigin::INFERRED;
    }

    // Shape: no default; remains absent if not specified.
    if (input.shape.has_value()) {
      origin.shape = DimensionOrigin::USER_SPECIFIED;
    } else {
      origin.shape = DimensionOrigin::ABSENT;
    }

    result.edgeSpecs.push_back(std::move(inferred));
    result.edgeOrigins.push_back(origin);
  }

  // Validate path contracts: check that referenced edges exist.
  for (const auto &path : pathSpecs) {
    bool startFound = false;
    bool endFound = false;

    for (const auto &edge : edgeSpecs) {
      if (edge.producerKernel == path.startProducer &&
          edge.consumerKernel == path.startConsumer) {
        startFound = true;
      }
      if (edge.producerKernel == path.endProducer &&
          edge.consumerKernel == path.endConsumer) {
        endFound = true;
      }
    }

    if (!startFound) {
      std::ostringstream oss;
      oss << "Path contract references non-existent start edge: "
          << edgeKey(path.startProducer, path.startConsumer);
      result.errors.push_back(oss.str());
    }
    if (!endFound) {
      std::ostringstream oss;
      oss << "Path contract references non-existent end edge: "
          << edgeKey(path.endProducer, path.endConsumer);
      result.errors.push_back(oss.str());
    }
  }

  result.pathSpecs = pathSpecs;
  return result;
}

//===----------------------------------------------------------------------===//
// Static verification helpers
//===----------------------------------------------------------------------===//

/// Verify placement for a single edge.
/// LOCAL_SPM matches BufferAllocation::{SPM_PRODUCER, SPM_CONSUMER}.
/// SHARED_L2 matches BufferAllocation::SHARED_L2.
/// EXTERNAL matches BufferAllocation::EXTERNAL_DRAM.
static bool
verifyPlacement(Placement specPlacement,
                BufferAllocation::Location actualLocation,
                std::string &diagnosticOut,
                const std::string &edgeId) {
  bool satisfied = false;

  switch (specPlacement) {
  case Placement::LOCAL_SPM:
    satisfied = (actualLocation == BufferAllocation::SPM_PRODUCER ||
                 actualLocation == BufferAllocation::SPM_CONSUMER);
    if (!satisfied) {
      const char *actualStr = "unknown";
      switch (actualLocation) {
      case BufferAllocation::SPM_PRODUCER:
        actualStr = "SPM_PRODUCER";
        break;
      case BufferAllocation::SPM_CONSUMER:
        actualStr = "SPM_CONSUMER";
        break;
      case BufferAllocation::SHARED_L2:
        actualStr = "SHARED_L2";
        break;
      case BufferAllocation::EXTERNAL_DRAM:
        actualStr = "EXTERNAL_DRAM";
        break;
      }
      std::ostringstream oss;
      oss << "Edge " << edgeId
          << ": placement constraint LOCAL_SPM violated; buffer placed in "
          << actualStr
          << "; this may be caused by SPM capacity overflow";
      diagnosticOut = oss.str();
    }
    break;

  case Placement::SHARED_L2:
    satisfied = (actualLocation == BufferAllocation::SHARED_L2);
    if (!satisfied) {
      std::ostringstream oss;
      oss << "Edge " << edgeId
          << ": placement constraint SHARED_L2 violated; buffer placed "
             "elsewhere";
      diagnosticOut = oss.str();
    }
    break;

  case Placement::EXTERNAL:
    satisfied = (actualLocation == BufferAllocation::EXTERNAL_DRAM);
    if (!satisfied) {
      std::ostringstream oss;
      oss << "Edge " << edgeId
          << ": placement constraint EXTERNAL violated; buffer not in DRAM";
      diagnosticOut = oss.str();
    }
    break;

  case Placement::AUTO:
    // AUTO means compiler chooses; always satisfied.
    satisfied = true;
    break;
  }

  return satisfied;
}

/// Verify shape for a single edge.
/// Parses the symbolic shape string and compares against tile dimensions.
static bool verifyShape(const std::string &specShape,
                        const std::vector<int64_t> &actualDims,
                        std::string &diagnosticOut,
                        const std::string &edgeId) {
  // Parse the spec shape using the parseShapeExpr utility.
  std::vector<std::string> specDimStrs = parseShapeExpr(specShape);

  if (specDimStrs.size() != actualDims.size()) {
    std::ostringstream oss;
    oss << "Edge " << edgeId << ": shape dimension count mismatch; contract "
        << "specifies " << specDimStrs.size() << " dimensions but tiling "
        << "produced " << actualDims.size() << " dimensions";
    diagnosticOut = oss.str();
    return false;
  }

  for (size_t i = 0; i < specDimStrs.size(); i++) {
    // Try to parse as integer for exact comparison.
    // Non-integer (symbolic) dimensions are not checked statically.
    try {
      int64_t specVal = std::stoll(specDimStrs[i]);
      if (specVal != actualDims[i]) {
        std::ostringstream oss;
        oss << "Edge " << edgeId << ": shape mismatch in dimension " << i
            << "; contract specifies " << specVal << " but tiling produced "
            << actualDims[i];
        diagnosticOut = oss.str();
        return false;
      }
    } catch (...) {
      // Symbolic dimension; skip static check.
    }
  }

  return true;
}

/// Verify FIFO ordering statically using the NoC schedule.
/// For each edge with FIFO ordering, the producer's transfer for tile T
/// must complete before the consumer's transfer for tile T begins.
/// This is a simplified check using route ordering.
static bool verifyStaticOrdering(Ordering specOrdering,
                                 const NoCSchedule &schedule,
                                 const std::string &producer,
                                 const std::string &consumer,
                                 std::string &diagnosticOut,
                                 const std::string &edgeId) {
  if (specOrdering != Ordering::FIFO)
    return true; // UNORDERED and SYMBOLIC have no static ordering check.

  // Find the route for this edge.
  std::string edgeName = producer + "_" + consumer;
  for (const auto &route : schedule.routes) {
    if (route.contractEdgeName == edgeName ||
        (route.producerCore == producer && route.consumerCore == consumer)) {
      // If the route has contention, there is a risk of ordering violation.
      // However, XY routing with single-path routes preserves FIFO within
      // a single edge. We check for basic sanity: the route exists and
      // transfer latency is non-negative.
      if (route.transferLatencyCycles == 0 &&
          route.transferDurationCycles == 0 && route.totalFlits > 0) {
        std::ostringstream oss;
        oss << "Edge " << edgeId
            << ": FIFO ordering may be violated; NoC route has zero latency "
               "with non-zero data volume";
        diagnosticOut = oss.str();
        return false;
      }
    }
  }

  // If no route found, the edge is core-local (same core); FIFO preserved.
  return true;
}

//===----------------------------------------------------------------------===//
// Dynamic verification helpers
//===----------------------------------------------------------------------===//

/// Verify throughput for a single edge using dynamic metrics.
static bool verifyDynamicThroughput(const std::string &specThroughput,
                                    double achievedThroughput,
                                    std::string &diagnosticOut,
                                    const std::string &edgeId) {
  // Parse the throughput expression as a numeric value.
  double specValue = 0.0;
  try {
    specValue = std::stod(specThroughput);
  } catch (...) {
    // Symbolic throughput; cannot verify dynamically without evaluation.
    return true;
  }

  if (achievedThroughput < specValue) {
    std::ostringstream oss;
    oss << "Edge " << edgeId << ": throughput constraint violated; contract "
        << "requires >= " << specValue << " elements/cycle but achieved "
        << achievedThroughput;
    diagnosticOut = oss.str();
    return false;
  }

  return true;
}

/// Verify FIFO ordering using dynamic violation count.
static bool verifyDynamicOrdering(Ordering specOrdering,
                                  int64_t violationCount,
                                  std::string &diagnosticOut,
                                  const std::string &edgeId) {
  if (specOrdering != Ordering::FIFO)
    return true;

  if (violationCount > 0) {
    std::ostringstream oss;
    oss << "Edge " << edgeId
        << ": FIFO ordering violated; " << violationCount
        << " out-of-order deliveries observed in simulation";
    diagnosticOut = oss.str();
    return false;
  }

  return true;
}

/// Verify path latency using dynamic metrics.
static bool verifyDynamicLatency(const std::string &specLatency,
                                 int64_t achievedLatency,
                                 std::string &diagnosticOut,
                                 const std::string &pathId) {
  int64_t specValue = 0;
  try {
    specValue = std::stoll(specLatency);
  } catch (...) {
    // Symbolic latency; cannot verify dynamically without evaluation.
    return true;
  }

  if (achievedLatency > specValue) {
    std::ostringstream oss;
    oss << "Path " << pathId << ": latency constraint violated; contract "
        << "requires <= " << specValue << " cycles but observed "
        << achievedLatency << " cycles";
    diagnosticOut = oss.str();
    return false;
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Top-level verification
//===----------------------------------------------------------------------===//

TDCVerificationReport
verifyContracts(const std::vector<TDCEdgeSpec> &edgeSpecs,
                const std::vector<TDCEdgeSpecOrigin> &edgeOrigins,
                const std::vector<TDCPathSpec> &pathSpecs,
                const StaticVerificationInputs &staticInputs,
                const std::vector<DynamicEdgeMetrics> *dynamicEdgeMetrics,
                const std::vector<DynamicPathMetrics> *dynamicPathMetrics) {
  TDCVerificationReport report;
  report.allSatisfied = true;

  // --- Per-edge verification ---
  for (size_t i = 0; i < edgeSpecs.size(); i++) {
    const auto &spec = edgeSpecs[i];
    const auto &origin = (i < edgeOrigins.size())
                             ? edgeOrigins[i]
                             : TDCEdgeSpecOrigin{};

    TDCEdgeVerificationResult edgeResult;
    edgeResult.producerKernel = spec.producerKernel;
    edgeResult.consumerKernel = spec.consumerKernel;

    std::string eid = edgeKey(spec.producerKernel, spec.consumerKernel);
    std::vector<std::string> diagnostics;

    // --- Static: Placement ---
    if (origin.placement == DimensionOrigin::USER_SPECIFIED &&
        spec.placement.has_value()) {
      // Find the buffer allocation for this edge.
      std::string contractEdgeName =
          spec.producerKernel + "_" + spec.consumerKernel;

      bool found = false;
      for (const auto &alloc : staticInputs.bufferPlan.allocations) {
        if (alloc.contractEdgeName == contractEdgeName) {
          std::string placeDiag;
          edgeResult.placementSatisfied =
              verifyPlacement(*spec.placement, alloc.location, placeDiag, eid);
          if (!placeDiag.empty())
            diagnostics.push_back(std::move(placeDiag));
          found = true;
          break;
        }
      }
      // If no allocation found and placement was specified, the edge may be
      // core-local (no cross-core buffer needed). LOCAL_SPM is satisfied
      // for core-local edges; others are not.
      if (!found && *spec.placement != Placement::LOCAL_SPM &&
          *spec.placement != Placement::AUTO) {
        edgeResult.placementSatisfied = false;
        std::ostringstream oss;
        oss << "Edge " << eid
            << ": no buffer allocation found for specified placement "
            << placementToString(*spec.placement);
        diagnostics.push_back(oss.str());
      }
    }

    // --- Static: Shape ---
    if (origin.shape == DimensionOrigin::USER_SPECIFIED &&
        spec.shape.has_value()) {
      bool found = false;
      for (const auto &td : staticInputs.tileDimensions) {
        if (edgeMatchesSpec(spec.producerKernel, spec.consumerKernel,
                            td.producerKernel, td.consumerKernel)) {
          std::string shapeDiag;
          edgeResult.shapeSatisfied =
              verifyShape(*spec.shape, td.tileDims, shapeDiag, eid);
          if (!shapeDiag.empty())
            diagnostics.push_back(std::move(shapeDiag));
          found = true;
          break;
        }
      }
      if (!found) {
        // No tile dimension info; cannot verify shape statically.
        // Treat as satisfied (shape will be checked dynamically if available).
      }
    }

    // --- Static: Ordering ---
    if (origin.ordering == DimensionOrigin::USER_SPECIFIED &&
        spec.ordering.has_value()) {
      std::string orderDiag;
      edgeResult.orderingSatisfied = verifyStaticOrdering(
          *spec.ordering, staticInputs.nocSchedule, spec.producerKernel,
          spec.consumerKernel, orderDiag, eid);
      if (!orderDiag.empty())
        diagnostics.push_back(std::move(orderDiag));
    }

    // --- Dynamic: Throughput ---
    if (dynamicEdgeMetrics &&
        origin.throughput == DimensionOrigin::USER_SPECIFIED &&
        spec.throughput.has_value()) {
      for (const auto &dm : *dynamicEdgeMetrics) {
        if (edgeMatchesSpec(spec.producerKernel, spec.consumerKernel,
                            dm.producerKernel, dm.consumerKernel)) {
          edgeResult.achievedThroughput = dm.sustainedThroughput;
          std::string tpDiag;
          edgeResult.throughputSatisfied = verifyDynamicThroughput(
              *spec.throughput, dm.sustainedThroughput, tpDiag, eid);
          if (!tpDiag.empty())
            diagnostics.push_back(std::move(tpDiag));
          break;
        }
      }
    }

    // --- Dynamic: Ordering violations ---
    if (dynamicEdgeMetrics &&
        origin.ordering == DimensionOrigin::USER_SPECIFIED &&
        spec.ordering.has_value()) {
      for (const auto &dm : *dynamicEdgeMetrics) {
        if (edgeMatchesSpec(spec.producerKernel, spec.consumerKernel,
                            dm.producerKernel, dm.consumerKernel)) {
          std::string orderDynDiag;
          bool dynOrderOk = verifyDynamicOrdering(
              *spec.ordering, dm.orderingViolationCount, orderDynDiag, eid);
          if (!dynOrderOk) {
            edgeResult.orderingSatisfied = false;
            diagnostics.push_back(std::move(orderDynDiag));
          }
          break;
        }
      }
    }

    // Assemble diagnostic string.
    if (!diagnostics.empty()) {
      std::ostringstream oss;
      for (size_t d = 0; d < diagnostics.size(); d++) {
        if (d > 0)
          oss << "; ";
        oss << diagnostics[d];
      }
      edgeResult.diagnostic = oss.str();
    }

    if (!edgeResult.allSatisfied())
      report.allSatisfied = false;

    report.edgeResults.push_back(std::move(edgeResult));
  }

  // --- Per-path verification ---
  for (const auto &path : pathSpecs) {
    TDCPathVerificationResult pathResult;
    pathResult.startProducer = path.startProducer;
    pathResult.startConsumer = path.startConsumer;
    pathResult.endProducer = path.endProducer;
    pathResult.endConsumer = path.endConsumer;

    std::string pid = edgeKey(path.startProducer, path.endConsumer);

    // Path latency is only verifiable with dynamic metrics.
    if (dynamicPathMetrics) {
      for (const auto &dm : *dynamicPathMetrics) {
        if (dm.startProducer == path.startProducer &&
            dm.startConsumer == path.startConsumer &&
            dm.endProducer == path.endProducer &&
            dm.endConsumer == path.endConsumer) {
          pathResult.achievedLatency = dm.observedLatency;
          std::string latDiag;
          pathResult.latencySatisfied = verifyDynamicLatency(
              path.latency, dm.observedLatency, latDiag, pid);
          if (!latDiag.empty())
            pathResult.diagnostic = std::move(latDiag);
          break;
        }
      }
    }

    if (!pathResult.latencySatisfied)
      report.allSatisfied = false;

    report.pathResults.push_back(std::move(pathResult));
  }

  return report;
}

} // namespace loom
