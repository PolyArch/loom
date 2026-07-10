#include "PnR/MappingEstimator.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <system_error>

llvm::Error
loom::pnr::writeMappingEstimateReportJson(llvm::StringRef outputPath,
                                          const MappingEstimateReport &report) {
  llvm::SmallString<256> parent(outputPath);
  llvm::sys::path::remove_filename(parent);
  if (!parent.empty()) {
    if (std::error_code ec = llvm::sys::fs::create_directories(parent))
      return llvm::createStringError(ec, "could not create %s", parent.c_str());
  }

  llvm::json::Array scoreBreakdown;
  scoreBreakdown.push_back(llvm::json::Object{
      {"category", "route_complexity"},
      {"score", static_cast<int64_t>(report.routeSegmentScore)},
      {"evidence", "mapping.route_segments"},
      {"heuristic", true},
      {"explanation", "one score unit per consumed route segment"},
  });
  scoreBreakdown.push_back(llvm::json::Object{
      {"category", "memory_resource"},
      {"score", static_cast<int64_t>(report.memoryAccessScore)},
      {"evidence", "fabric.mem placement"},
      {"heuristic", true},
      {"explanation",
       "weighted mapped load and store resources, with additional store "
       "commit pressure"},
  });
  scoreBreakdown.push_back(llvm::json::Object{
      {"category", "width_adapter"},
      {"score", static_cast<int64_t>(report.widthAdapterScore)},
      {"evidence", "mapping.placements.operation"},
      {"heuristic", true},
      {"explanation", "mapped truncation and zero-extension resources"},
  });
  scoreBreakdown.push_back(llvm::json::Object{
      {"category", "expensive_functional_unit"},
      {"score", static_cast<int64_t>(report.functionalUnitScore)},
      {"evidence", "mapping.placements.operation"},
      {"heuristic", true},
      {"explanation",
       "mapped multiply, divide, remainder, and fused multiply-add resources"},
  });
  scoreBreakdown.push_back(llvm::json::Object{
      {"category", "resource_diversity"},
      {"score", static_cast<int64_t>(report.resourceMixScore)},
      {"evidence", "mapping.placements.operation"},
      {"heuristic", true},
      {"explanation", "number of distinct mapped operation kinds"},
  });
  scoreBreakdown.push_back(llvm::json::Object{
      {"category", "computed_load_address"},
      {"score", static_cast<int64_t>(report.loadAddressScore)},
      {"evidence", "mapping.routes.edge_ref"},
      {"heuristic", true},
      {"explanation", "computed SSA values feeding fabric load-address ports"},
  });
  scoreBreakdown.push_back(llvm::json::Object{
      {"category", "computed_store_address"},
      {"score", static_cast<int64_t>(report.storeAddressScore)},
      {"evidence", "mapping.routes.edge_ref"},
      {"heuristic", true},
      {"explanation", "computed SSA values feeding fabric store-address ports"},
  });
  scoreBreakdown.push_back(llvm::json::Object{
      {"category", "configuration_volume"},
      {"score", static_cast<int64_t>(report.configLoadScore)},
      {"evidence", "mapping.config_records"},
      {"heuristic", true},
      {"explanation", "one score unit per 128 configuration records"},
  });
  scoreBreakdown.push_back(llvm::json::Object{
      {"category", "temporal_sharing"},
      {"score", static_cast<int64_t>(report.temporalConflictScore)},
      {"evidence", "placement schedule"},
      {"heuristic", true},
      {"explanation", "temporal placements weighted by routed-edge count"},
  });

  llvm::json::Array limitations;
  limitations.push_back("not_a_timing_model");
  limitations.push_back("not_a_functional_simulator");
  limitations.push_back("no_route_congestion_model");
  limitations.push_back("no_fifo_queueing_model");
  limitations.push_back("no_memory_bank_conflict_model");

  llvm::json::Object root{
      {"schema_version", "1.0"},
      {"kind", "mapping_estimate_report"},
      {"workload", report.workload},
      {"hardware", report.hardware},
      {"mapping_id", report.mappingId},
      {"config_id", report.configId},
      {"config_fingerprint", report.configFingerprint},
      {"component_config_view", report.componentConfigView},
      {"component_config_fingerprint", report.componentConfigFingerprint},
      {"status", report.status},
      {"fidelity_level", "static_mapping_heuristic"},
      {"metric_definition", "weighted_mapping_artifact_counts.v1"},
      {"placed_records", static_cast<int64_t>(report.placedRecords)},
      {"routed_edges", static_cast<int64_t>(report.routedEdges)},
      {"route_segments", static_cast<int64_t>(report.routeSegments)},
      {"config_records", static_cast<int64_t>(report.configRecords)},
      {"spatial_placements", static_cast<int64_t>(report.spatialPlacements)},
      {"temporal_placements", static_cast<int64_t>(report.temporalPlacements)},
      {"limitations", std::move(limitations)},
  };
  if (report.status == "pass") {
    root["route_segment_score"] = report.routeSegmentScore;
    root["memory_access_score"] = report.memoryAccessScore;
    root["width_adapter_score"] = report.widthAdapterScore;
    root["functional_unit_score"] = report.functionalUnitScore;
    root["resource_mix_score"] = report.resourceMixScore;
    root["load_address_score"] = report.loadAddressScore;
    root["store_address_score"] = report.storeAddressScore;
    root["config_load_score"] = report.configLoadScore;
    root["temporal_conflict_score"] = report.temporalConflictScore;
    root["total_cost_score"] = report.totalCostScore;
    root["score_breakdown"] = std::move(scoreBreakdown);
  }
  if (!report.diagnostic.empty()) {
    llvm::json::Array diagnostics;
    diagnostics.push_back(report.diagnostic);
    root.try_emplace("diagnostics", std::move(diagnostics));
  }
  if (!report.hardwareArtifact.empty())
    root.try_emplace("hardware_artifact", report.hardwareArtifact);

  std::error_code ec;
  llvm::raw_fd_ostream out(outputPath, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return llvm::createStringError(ec, "could not open %s",
                                   outputPath.str().c_str());
  out << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << '\n';
  return llvm::Error::success();
}
