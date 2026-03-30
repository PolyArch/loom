#include "loom/SystemCompiler/TDCVerification.h"

#include <algorithm>
#include <sstream>

namespace loom {

//===----------------------------------------------------------------------===//
// TDCVerificationResult helpers (legacy structural API)
//===----------------------------------------------------------------------===//

void TDCVerificationResult::addWarning(const std::string &msg) {
  diagnostics.push_back({TDCDiagnostic::Severity::Warning, msg});
}

void TDCVerificationResult::addError(const std::string &msg) {
  valid = false;
  diagnostics.push_back({TDCDiagnostic::Severity::Error, msg});
}

//===----------------------------------------------------------------------===//
// Legacy structural verification
//===----------------------------------------------------------------------===//

TDCVerificationResult verifyEdgeSpec(const TDCEdgeSpec &spec) {
  TDCVerificationResult result;

  if (spec.producerKernel.empty())
    result.addError("TDCEdgeSpec: producerKernel is empty");
  if (spec.consumerKernel.empty())
    result.addError("TDCEdgeSpec: consumerKernel is empty");
  if (spec.dataTypeName.empty())
    result.addError("TDCEdgeSpec: dataTypeName is empty");

  if (spec.shape.has_value() && !spec.shape->empty()) {
    auto dims = parseShapeExpr(*spec.shape);
    if (dims.empty())
      result.addWarning("TDCEdgeSpec: shape '" + *spec.shape +
                        "' parsed to zero dimensions");
  }

  if (spec.throughput.has_value() && spec.throughput->empty())
    result.addWarning(
        "TDCEdgeSpec: throughput is set but empty for edge " +
        spec.producerKernel + "->" + spec.consumerKernel);

  return result;
}

TDCVerificationResult verifyPathSpec(const TDCPathSpec &spec) {
  TDCVerificationResult result;

  if (spec.startProducer.empty())
    result.addError("TDCPathSpec: startProducer is empty");
  if (spec.startConsumer.empty())
    result.addError("TDCPathSpec: startConsumer is empty");
  if (spec.endProducer.empty())
    result.addError("TDCPathSpec: endProducer is empty");
  if (spec.endConsumer.empty())
    result.addError("TDCPathSpec: endConsumer is empty");
  if (spec.latency.empty())
    result.addError("TDCPathSpec: latency expression is empty");

  return result;
}

TDCVerificationResult
verifyContractsStructural(const std::vector<TDCEdgeSpec> &edges,
                          const std::vector<TDCPathSpec> &paths) {
  TDCVerificationResult merged;

  for (const auto &edge : edges) {
    auto r = verifyEdgeSpec(edge);
    if (!r.valid)
      merged.valid = false;
    merged.diagnostics.insert(merged.diagnostics.end(),
                              r.diagnostics.begin(), r.diagnostics.end());
  }

  for (const auto &path : paths) {
    auto r = verifyPathSpec(path);
    if (!r.valid)
      merged.valid = false;
    merged.diagnostics.insert(merged.diagnostics.end(),
                              r.diagnostics.begin(), r.diagnostics.end());
  }

  return merged;
}

//===----------------------------------------------------------------------===//
// Contract inference
//===----------------------------------------------------------------------===//

InferenceResult
inferEdgeContracts(const std::vector<TDCEdgeSpec> &edgeSpecs) {
  InferenceResult result;
  result.edgeSpecs.reserve(edgeSpecs.size());
  result.origins.reserve(edgeSpecs.size());

  for (const auto &spec : edgeSpecs) {
    TDCEdgeSpec inferred = spec; // copy
    TDCEdgeSpecOrigin origin;

    // Ordering: fill missing with FIFO (conservative default)
    if (spec.ordering.has_value()) {
      origin.ordering = DimensionOrigin::USER_SPECIFIED;
    } else {
      inferred.ordering = Ordering::FIFO;
      origin.ordering = DimensionOrigin::INFERRED;
    }

    // Throughput: no default, remains absent
    if (spec.throughput.has_value()) {
      origin.throughput = DimensionOrigin::USER_SPECIFIED;
    } else {
      origin.throughput = DimensionOrigin::ABSENT;
    }

    // Placement: fill missing with AUTO (compiler decides)
    if (spec.placement.has_value()) {
      origin.placement = DimensionOrigin::USER_SPECIFIED;
    } else {
      inferred.placement = Placement::AUTO;
      origin.placement = DimensionOrigin::INFERRED;
    }

    // Shape: no default, remains absent
    if (spec.shape.has_value()) {
      origin.shape = DimensionOrigin::USER_SPECIFIED;
    } else {
      origin.shape = DimensionOrigin::ABSENT;
    }

    result.edgeSpecs.push_back(std::move(inferred));
    result.origins.push_back(origin);
  }

  return result;
}

std::vector<std::string>
validatePathReferences(const std::vector<TDCPathSpec> &pathSpecs,
                       const std::vector<TDCEdgeSpec> &edgeSpecs) {
  std::vector<std::string> errors;

  // Build a set of known (producer, consumer) edge identities.
  std::vector<std::pair<std::string, std::string>> knownEdges;
  for (const auto &e : edgeSpecs)
    knownEdges.emplace_back(e.producerKernel, e.consumerKernel);

  auto edgeExists = [&](const std::string &prod,
                         const std::string &cons) -> bool {
    for (const auto &kv : knownEdges) {
      if (kv.first == prod && kv.second == cons)
        return true;
    }
    return false;
  };

  for (const auto &path : pathSpecs) {
    if (!edgeExists(path.startProducer, path.startConsumer)) {
      errors.push_back(
          "TDCPathSpec: start edge (" + path.startProducer + " -> " +
          path.startConsumer + ") does not exist in edge specs");
    }
    if (!edgeExists(path.endProducer, path.endConsumer)) {
      errors.push_back(
          "TDCPathSpec: end edge (" + path.endProducer + " -> " +
          path.endConsumer + ") does not exist in edge specs");
    }
  }

  return errors;
}

//===----------------------------------------------------------------------===//
// Helpers for matching edges in compile-time outputs
//===----------------------------------------------------------------------===//

namespace {

/// Find a BufferAllocation by edge name (producer->consumer).
const BufferAllocation *
findAllocation(const BufferAllocationPlan &plan,
               const std::string &producer,
               const std::string &consumer) {
  std::string edgeName = producer + "->" + consumer;
  for (const auto &alloc : plan.allocations) {
    if (alloc.contractEdgeName == edgeName)
      return &alloc;
  }
  return nullptr;
}

/// Find tile dimensions for an edge.
const EdgeTileDimensions *
findTileDims(const std::vector<EdgeTileDimensions> &tileDims,
             const std::string &producer,
             const std::string &consumer) {
  for (const auto &td : tileDims) {
    if (td.producerKernel == producer && td.consumerKernel == consumer)
      return &td;
  }
  return nullptr;
}

/// Find scheduling slot for an edge.
const EdgeSchedulingSlot *
findSchedSlot(const std::vector<EdgeSchedulingSlot> &slots,
              const std::string &producer,
              const std::string &consumer) {
  for (const auto &s : slots) {
    if (s.producerKernel == producer && s.consumerKernel == consumer)
      return &s;
  }
  return nullptr;
}

/// Find dynamic edge metrics.
const DynamicEdgeMetrics *
findEdgeMetrics(const std::vector<DynamicEdgeMetrics> &metrics,
                const std::string &producer,
                const std::string &consumer) {
  for (const auto &m : metrics) {
    if (m.producerKernel == producer && m.consumerKernel == consumer)
      return &m;
  }
  return nullptr;
}

/// Find dynamic path metrics.
const DynamicPathMetrics *
findPathMetrics(const std::vector<DynamicPathMetrics> &metrics,
                const std::string &startP, const std::string &startC,
                const std::string &endP, const std::string &endC) {
  for (const auto &m : metrics) {
    if (m.startProducer == startP && m.startConsumer == startC &&
        m.endProducer == endP && m.endConsumer == endC)
      return &m;
  }
  return nullptr;
}

/// Check whether a Placement matches a BufferAllocation::Location.
/// LOCAL_SPM matches SPM_PRODUCER or SPM_CONSUMER.
/// SHARED_L2 matches SHARED_L2.
/// EXTERNAL matches EXTERNAL_DRAM.
bool placementMatchesLocation(Placement p, BufferAllocation::Location loc) {
  switch (p) {
  case Placement::LOCAL_SPM:
    return loc == BufferAllocation::SPM_PRODUCER ||
           loc == BufferAllocation::SPM_CONSUMER;
  case Placement::SHARED_L2:
    return loc == BufferAllocation::SHARED_L2;
  case Placement::EXTERNAL:
    return loc == BufferAllocation::EXTERNAL_DRAM;
  case Placement::AUTO:
    return true; // AUTO always matches
  }
  return true;
}

std::string locationToString(BufferAllocation::Location loc) {
  switch (loc) {
  case BufferAllocation::SPM_PRODUCER: return "SPM_PRODUCER";
  case BufferAllocation::SPM_CONSUMER: return "SPM_CONSUMER";
  case BufferAllocation::SHARED_L2: return "SHARED_L2";
  case BufferAllocation::EXTERNAL_DRAM: return "EXTERNAL_DRAM";
  }
  return "UNKNOWN";
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Static verification
//===----------------------------------------------------------------------===//

TDCVerificationReport
verifyStatic(const std::vector<TDCEdgeSpec> &edgeSpecs,
             const std::vector<TDCEdgeSpecOrigin> &origins,
             const BufferAllocationPlan &bufferPlan,
             const std::vector<EdgeTileDimensions> &tileDims,
             const std::vector<EdgeSchedulingSlot> &schedSlots,
             const std::map<std::string, int64_t> &params) {
  TDCVerificationReport report;

  for (size_t iter_var0 = 0; iter_var0 < edgeSpecs.size(); ++iter_var0) {
    const auto &spec = edgeSpecs[iter_var0];
    const auto &origin = origins[iter_var0];

    TDCEdgeVerificationResult edgeResult;
    edgeResult.producerKernel = spec.producerKernel;
    edgeResult.consumerKernel = spec.consumerKernel;

    // --- Placement check (static) ---
    if (origin.placement == DimensionOrigin::USER_SPECIFIED &&
        spec.placement.has_value() &&
        *spec.placement != Placement::AUTO) {
      const auto *alloc =
          findAllocation(bufferPlan, spec.producerKernel, spec.consumerKernel);
      if (alloc) {
        if (!placementMatchesLocation(*spec.placement, alloc->location)) {
          edgeResult.placementSatisfied = false;
          std::string diag =
              "buffer for edge " + spec.producerKernel + " -> " +
              spec.consumerKernel + " was placed in " +
              locationToString(alloc->location) +
              " but contract requires " +
              placementToString(*spec.placement);
          // Add suggestion
          if (*spec.placement == Placement::LOCAL_SPM &&
              (alloc->location == BufferAllocation::SHARED_L2 ||
               alloc->location == BufferAllocation::EXTERNAL_DRAM)) {
            diag += "; this may be caused by SPM capacity overflow";
          }
          edgeResult.diagnostics.push_back(diag);
        }
      }
      // If no allocation found, skip (edge might be co-located)
    }

    // --- Shape check (static) ---
    if (origin.shape == DimensionOrigin::USER_SPECIFIED &&
        spec.shape.has_value()) {
      const auto *td =
          findTileDims(tileDims, spec.producerKernel, spec.consumerKernel);
      if (td) {
        // Parse the specified shape to get concrete dimensions
        auto specDimStrings = parseShapeExpr(*spec.shape);
        // Resolve symbolic dimension expressions using the provided params.
        std::vector<int64_t> specDims;
        bool parseOk = true;
        for (const auto &ds : specDimStrings) {
          auto evalResult = evaluateSymbolicExpr(ds, params);
          if (evalResult.ok()) {
            specDims.push_back(*evalResult.value);
          } else {
            parseOk = false;
            break;
          }
        }

        if (parseOk) {
          if (specDims != td->dimensions) {
            edgeResult.shapeSatisfied = false;
            std::ostringstream diag;
            diag << "shape mismatch for edge " << spec.producerKernel
                 << " -> " << spec.consumerKernel << ": contract requires [";
            for (size_t iter_var1 = 0; iter_var1 < specDims.size();
                 ++iter_var1) {
              if (iter_var1 > 0)
                diag << ", ";
              diag << specDims[iter_var1];
            }
            diag << "] but tiling engine produced [";
            for (size_t iter_var1 = 0; iter_var1 < td->dimensions.size();
                 ++iter_var1) {
              if (iter_var1 > 0)
                diag << ", ";
              diag << td->dimensions[iter_var1];
            }
            diag << "]";
            edgeResult.diagnostics.push_back(diag.str());
          }
        }
        // If shape is symbolic and we cannot resolve it without params, skip
      }
    }

    // --- Ordering check (static) ---
    if (origin.ordering == DimensionOrigin::USER_SPECIFIED &&
        spec.ordering.has_value() &&
        *spec.ordering == Ordering::FIFO) {
      const auto *slot =
          findSchedSlot(schedSlots, spec.producerKernel, spec.consumerKernel);
      if (slot) {
        // Check that for each tile T, producer completes before consumer starts
        size_t numTiles = std::min(slot->producerCompletionTimes.size(),
                                   slot->consumerStartTimes.size());
        for (size_t iter_var1 = 0; iter_var1 < numTiles; ++iter_var1) {
          if (slot->producerCompletionTimes[iter_var1] >
              slot->consumerStartTimes[iter_var1]) {
            edgeResult.orderingSatisfied = false;
            std::ostringstream diag;
            diag << "FIFO ordering violated for edge "
                 << spec.producerKernel << " -> " << spec.consumerKernel
                 << " at tile " << iter_var1
                 << ": producer completes at cycle "
                 << slot->producerCompletionTimes[iter_var1]
                 << " but consumer starts at cycle "
                 << slot->consumerStartTimes[iter_var1];
            edgeResult.diagnostics.push_back(diag.str());
            break; // Report first violation only
          }
        }
      }
    }

    if (!edgeResult.orderingSatisfied || !edgeResult.placementSatisfied ||
        !edgeResult.shapeSatisfied) {
      report.allSatisfied = false;
    }

    report.edgeResults.push_back(std::move(edgeResult));
  }

  return report;
}

//===----------------------------------------------------------------------===//
// Dynamic verification
//===----------------------------------------------------------------------===//

TDCVerificationReport
verifyDynamic(const std::vector<TDCEdgeSpec> &edgeSpecs,
              const std::vector<TDCEdgeSpecOrigin> &origins,
              const std::vector<TDCPathSpec> &pathSpecs,
              const std::vector<DynamicEdgeMetrics> &edgeMetrics,
              const std::vector<DynamicPathMetrics> &pathMetrics,
              const std::map<std::string, int64_t> &params) {
  TDCVerificationReport report;

  // Edge-level dynamic checks
  for (size_t iter_var0 = 0; iter_var0 < edgeSpecs.size(); ++iter_var0) {
    const auto &spec = edgeSpecs[iter_var0];
    const auto &origin = origins[iter_var0];

    TDCEdgeVerificationResult edgeResult;
    edgeResult.producerKernel = spec.producerKernel;
    edgeResult.consumerKernel = spec.consumerKernel;

    const auto *metrics =
        findEdgeMetrics(edgeMetrics, spec.producerKernel, spec.consumerKernel);

    // --- Throughput check (dynamic) ---
    if (origin.throughput == DimensionOrigin::USER_SPECIFIED &&
        spec.throughput.has_value() && metrics) {
      auto evalResult = evaluateSymbolicExpr(*spec.throughput, params);
      if (evalResult.ok()) {
        double required = static_cast<double>(*evalResult.value);
        edgeResult.achievedThroughput = metrics->sustainedThroughput;
        if (metrics->sustainedThroughput < required) {
          edgeResult.throughputSatisfied = false;
          std::ostringstream diag;
          diag << "throughput violation for edge " << spec.producerKernel
               << " -> " << spec.consumerKernel
               << ": contract requires >= " << required
               << " elements/cycle but achieved "
               << metrics->sustainedThroughput;
          edgeResult.diagnostics.push_back(diag.str());
        }
      }
    }

    // --- Ordering check (dynamic) ---
    if (origin.ordering == DimensionOrigin::USER_SPECIFIED &&
        spec.ordering.has_value() &&
        *spec.ordering == Ordering::FIFO && metrics) {
      if (metrics->orderingViolationCount > 0) {
        edgeResult.orderingSatisfied = false;
        std::ostringstream diag;
        diag << "FIFO ordering violated dynamically for edge "
             << spec.producerKernel << " -> " << spec.consumerKernel
             << ": " << metrics->orderingViolationCount
             << " out-of-order deliveries observed";
        edgeResult.diagnostics.push_back(diag.str());
      }
    }

    if (!edgeResult.throughputSatisfied || !edgeResult.orderingSatisfied) {
      report.allSatisfied = false;
    }

    report.edgeResults.push_back(std::move(edgeResult));
  }

  // Path-level dynamic checks
  for (const auto &pathSpec : pathSpecs) {
    TDCPathVerificationResult pathResult;
    pathResult.startProducer = pathSpec.startProducer;
    pathResult.startConsumer = pathSpec.startConsumer;
    pathResult.endProducer = pathSpec.endProducer;
    pathResult.endConsumer = pathSpec.endConsumer;

    const auto *metrics =
        findPathMetrics(pathMetrics, pathSpec.startProducer,
                        pathSpec.startConsumer, pathSpec.endProducer,
                        pathSpec.endConsumer);
    if (metrics) {
      auto evalResult = evaluateSymbolicExpr(pathSpec.latency, params);
      if (evalResult.ok()) {
        int64_t maxCycles = *evalResult.value;
        pathResult.achievedLatency = metrics->observedLatency;
        if (metrics->observedLatency > maxCycles) {
          pathResult.latencySatisfied = false;
          std::ostringstream diag;
          diag << "path latency violation (" << pathSpec.startProducer
               << " -> " << pathSpec.startConsumer << " ... "
               << pathSpec.endProducer << " -> " << pathSpec.endConsumer
               << "): contract requires <= " << maxCycles
               << " cycles but observed " << metrics->observedLatency;
          pathResult.diagnostics.push_back(diag.str());
        }
      }
    }

    if (!pathResult.latencySatisfied) {
      report.allSatisfied = false;
    }

    report.pathResults.push_back(std::move(pathResult));
  }

  return report;
}

//===----------------------------------------------------------------------===//
// Top-level verification entry point
//===----------------------------------------------------------------------===//

TDCVerificationReport
verifyContracts(
    const std::vector<TDCEdgeSpec> &edgeSpecs,
    const std::vector<TDCEdgeSpecOrigin> &origins,
    const std::vector<TDCPathSpec> &pathSpecs,
    const BufferAllocationPlan &bufferPlan,
    const std::vector<EdgeTileDimensions> &tileDims,
    const std::vector<EdgeSchedulingSlot> &schedSlots,
    const std::optional<std::vector<DynamicEdgeMetrics>> &edgeMetrics,
    const std::optional<std::vector<DynamicPathMetrics>> &pathMetrics,
    const std::map<std::string, int64_t> &params) {

  // Always run static verification, forwarding params for symbolic shape eval
  TDCVerificationReport report =
      verifyStatic(edgeSpecs, origins, bufferPlan, tileDims, schedSlots, params);

  // Run dynamic verification when metrics are provided
  if (edgeMetrics.has_value() || pathMetrics.has_value()) {
    std::vector<DynamicEdgeMetrics> em =
        edgeMetrics.value_or(std::vector<DynamicEdgeMetrics>{});
    std::vector<DynamicPathMetrics> pm =
        pathMetrics.value_or(std::vector<DynamicPathMetrics>{});

    TDCVerificationReport dynReport =
        verifyDynamic(edgeSpecs, origins, pathSpecs, em, pm, params);

    // Merge dynamic results into the static report.
    // Edge results: overlay dynamic fields onto matching static results.
    if (dynReport.edgeResults.size() == report.edgeResults.size()) {
      for (size_t iter_var0 = 0; iter_var0 < report.edgeResults.size();
           ++iter_var0) {
        auto &dst = report.edgeResults[iter_var0];
        const auto &src = dynReport.edgeResults[iter_var0];
        if (!src.throughputSatisfied) {
          dst.throughputSatisfied = false;
          report.allSatisfied = false;
        }
        if (!src.orderingSatisfied) {
          dst.orderingSatisfied = false;
          report.allSatisfied = false;
        }
        dst.achievedThroughput = src.achievedThroughput;
        dst.diagnostics.insert(dst.diagnostics.end(),
                               src.diagnostics.begin(),
                               src.diagnostics.end());
      }
    }

    // Path results come only from dynamic verification.
    report.pathResults = std::move(dynReport.pathResults);
    if (!dynReport.allSatisfied)
      report.allSatisfied = false;
  }

  return report;
}

} // namespace loom
