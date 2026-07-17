#include "MappingInternal.h"

#include "Dataflow/IR/DataflowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <set>
#include <string>
#include <system_error>

using namespace loom::pnr;
using namespace loom::pnr::detail;

namespace {

std::string csvEscape(llvm::StringRef value) {
  if (value.find_first_of(",\"\n\r") == llvm::StringRef::npos)
    return value.str();
  std::string escaped = "\"";
  for (char ch : value) {
    if (ch == '"')
      escaped.push_back('"');
    escaped.push_back(ch);
  }
  escaped.push_back('"');
  return escaped;
}

llvm::Error createParentDirectories(llvm::StringRef outputPath) {
  llvm::SmallString<256> parent(outputPath);
  llvm::sys::path::remove_filename(parent);
  if (parent.empty())
    return llvm::Error::success();
  if (std::error_code ec = llvm::sys::fs::create_directories(parent))
    return llvm::createStringError(ec, "could not create %s", parent.c_str());
  return llvm::Error::success();
}

} // namespace

namespace loom::pnr::detail {

namespace {

llvm::json::Object placementJson(const PlacementRecord &placement) {
  return llvm::json::Object{
      {"software", placement.softwareId},
      {"operation", placement.operation},
      {"resource_kind", placement.resourceKind},
      {"hardware", placement.hardwareId},
      {"schedule", placement.schedule},
  };
}

llvm::json::Object routeJson(const RouteRecord &route) {
  llvm::json::Array segments;
  for (const RouteSegment &segment : route.segments) {
    llvm::json::Object segmentObject{
        {"segment_id", segment.segmentId},
        {"segment_kind", segment.segmentKind},
        {"source_endpoint", segment.sourceEndpoint},
        {"sink_endpoint", segment.sinkEndpoint},
    };
    if (!segment.hardwareRef.empty())
      segmentObject.try_emplace("hardware_ref", segment.hardwareRef);
    segments.push_back(std::move(segmentObject));
  }
  return llvm::json::Object{
      {"record_id", route.recordId},
      {"edge_ref", route.edgeRef},
      {"producer_binding", route.producerBinding},
      {"consumer_binding", route.consumerBinding},
      {"payload_kind", route.payloadKind},
      {"from", route.fromSoftwareId},
      {"to", route.toSoftwareId},
      {"status", "routed"},
      {"segments", std::move(segments)},
  };
}

llvm::json::Object unroutedEdgeJson(const UnroutedEdgeRecord &edge) {
  return llvm::json::Object{
      {"edge_ref", edge.edgeRef},
      {"producer_binding", edge.producerBinding},
      {"consumer_binding", edge.consumerBinding},
      {"payload_kind", edge.payloadKind},
      {"from", edge.fromSoftwareId},
      {"to", edge.toSoftwareId},
      {"status", "unrouted"},
      {"source_endpoint", edge.sourceEndpoint},
      {"sink_endpoint", edge.sinkEndpoint},
      {"diagnostic", edge.diagnostic},
  };
}

void addConfig(llvm::SmallVectorImpl<ConfigEntry> &entries,
               llvm::StringRef target, llvm::StringRef registerName,
               llvm::StringRef value, llvm::StringRef source) {
  entries.push_back(
      ConfigEntry{target.str(), registerName.str(), value.str(), source.str()});
}

} // namespace

llvm::Error appendPlacementConfig(MappingSummary &summary,
                                  const SoftwareNode &node,
                                  const HardwareResource &resource) {
  std::set<std::string> emittedSwConfigKeys;
  if (resource.kind == ResourceKind::FabricOp) {
    if (std::optional<std::string> opSel = configFor(resource, "op_sel")) {
      if (*opSel != node.operation)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "hardware resource %s is configured for %s but software op is %s",
            resource.id.c_str(), opSel->c_str(), node.operation.c_str());
    }
  }

  std::string source = "placement:" + node.id;
  addConfig(summary.configEntries, resource.id, "software_id", node.id, source);
  addConfig(summary.configEntries, resource.id, "operation", node.operation,
            source);
  addConfig(summary.configEntries, resource.id, "resource_kind",
            resourceKindName(node.resourceKind), source);
  addConfig(summary.configEntries, resource.id, "schedule", resource.schedule,
            source);

  if (resource.kind == ResourceKind::FabricOp &&
      resource.supportedOps.size() > 1 && !configFor(resource, "op_sel")) {
    addConfig(summary.configEntries, resource.id, "sw_configs.op_sel",
              node.operation, source);
    emittedSwConfigKeys.insert("op_sel");
  }
  if (auto stream = mlir::dyn_cast_or_null<dataflow::StreamOp>(node.op)) {
    const auto &config = resource.streamConfiguration;
    std::string predicate =
        mlir::arith::stringifyCmpIPredicate(stream.getPredicate()).str();
    if (!config || config->stepKind != stream.getStepKind() ||
        !config->supports(stream.getPredicate()) ||
        (config->selectedPredicate &&
         *config->selectedPredicate != stream.getPredicate()))
      return llvm::createStringError(
          std::errc::invalid_argument,
          "hardware resource %s does not support software config "
          "predicate=%s",
          resource.id.c_str(), predicate.c_str());
    addConfig(summary.configEntries, resource.id, "sw_configs.predicate",
              predicate, source);
    emittedSwConfigKeys.insert("predicate");
  }
  for (const auto &[key, value] : softwareConfigsFor(node)) {
    std::optional<std::string> resolvedValue =
        resolvedSoftwareConfigValue(resource, key, value);
    if (!resolvedValue)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "hardware resource %s does not support software config %s=%s",
          resource.id.c_str(), key.c_str(), value.c_str());
    addConfig(summary.configEntries, resource.id, "sw_configs." + key,
              *resolvedValue, source);
    emittedSwConfigKeys.insert(key);
  }
  for (const auto &[key, value] : resource.swConfigs) {
    if (emittedSwConfigKeys.count(key) != 0)
      continue;
    addConfig(summary.configEntries, resource.id, "sw_configs." + key, value,
              source);
  }
  return llvm::Error::success();
}

namespace {

std::string routeTarget(const MappingSummary &summary,
                        llvm::StringRef recordId) {
  return summary.mappingId + "::" + recordId.str();
}

std::string routeSource(const RouteRecord &route) {
  return "route:" + route.recordId;
}

} // namespace

void appendRouteConfig(MappingSummary &summary) {
  for (const RouteRecord &route : summary.routes) {
    std::string source = routeSource(route);
    std::string target = routeTarget(summary, route.recordId);
    addConfig(summary.configEntries, target, "from_software_id",
              route.fromSoftwareId, source);
    addConfig(summary.configEntries, target, "to_software_id",
              route.toSoftwareId, source);
    addConfig(summary.configEntries, target, "segment_count",
              std::to_string(route.segments.size()), source);
    for (std::size_t segmentIndex = 0; segmentIndex < route.segments.size();
         ++segmentIndex) {
      const RouteSegment &segment = route.segments[segmentIndex];
      std::string prefix = "segment." + std::to_string(segmentIndex) + ".";
      addConfig(summary.configEntries, target, prefix + "kind",
                segment.segmentKind, source);
      addConfig(summary.configEntries, target, prefix + "source_endpoint",
                segment.sourceEndpoint, source);
      addConfig(summary.configEntries, target, prefix + "sink_endpoint",
                segment.sinkEndpoint, source);
      if (!segment.hardwareRef.empty())
        addConfig(summary.configEntries, target, prefix + "hardware_ref",
                  segment.hardwareRef, source);
    }
  }
}

namespace {

std::string configKey(llvm::StringRef target, llvm::StringRef registerName,
                      llvm::StringRef source) {
  std::string key;
  llvm::raw_string_ostream os(key);
  os << target << '\x1f' << registerName << '\x1f' << source;
  return key;
}

std::string registerKey(llvm::StringRef target, llvm::StringRef registerName) {
  std::string key;
  llvm::raw_string_ostream os(key);
  os << target << '\x1f' << registerName;
  return key;
}

} // namespace

llvm::Error validateConfigBitstream(const MappingSummary &summary) {
  if (summary.status != "pass")
    return llvm::Error::success();

  llvm::StringSet<> seen;
  llvm::StringSet<> writtenRegisters;
  for (const ConfigEntry &entry : summary.configEntries) {
    if (entry.target.empty() || entry.registerName.empty() ||
        entry.source.empty())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "config bitstream contains an incomplete "
                                     "register assignment");
    if (entry.registerName == "schedule" && entry.value != "spatial" &&
        entry.value != "temporal")
      return llvm::createStringError(std::errc::invalid_argument,
                                     "config bitstream contains invalid "
                                     "schedule value %s",
                                     entry.value.c_str());
    std::string key = configKey(entry.target, entry.registerName, entry.source);
    if (!seen.insert(key).second)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "config bitstream contains duplicate assignment for %s",
          entry.target.c_str());
    std::string regKey = registerKey(entry.target, entry.registerName);
    if (!writtenRegisters.insert(regKey).second)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "config bitstream writes register %s on %s more than once",
          entry.registerName.c_str(), entry.target.c_str());
  }

  for (const PlacementRecord &placement : summary.placements) {
    std::string source = "placement:" + placement.softwareId;
    for (llvm::StringRef reg :
         {"software_id", "operation", "resource_kind", "schedule"}) {
      if (!seen.contains(configKey(placement.hardwareId, reg, source)))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "config bitstream is missing placement register %s for %s",
            reg.str().c_str(), placement.hardwareId.c_str());
    }
  }

  for (const RouteRecord &route : summary.routes) {
    std::string source = routeSource(route);
    std::string target = routeTarget(summary, route.recordId);
    if (!seen.contains(configKey(target, "from_software_id", source)) ||
        !seen.contains(configKey(target, "to_software_id", source)) ||
        !seen.contains(configKey(target, "segment_count", source)))
      return llvm::createStringError(
          std::errc::invalid_argument,
          "config bitstream is missing route endpoint registers for %s",
          target.c_str());
    for (std::size_t segmentIndex = 0; segmentIndex < route.segments.size();
         ++segmentIndex) {
      std::string prefix = "segment." + std::to_string(segmentIndex) + ".";
      for (llvm::StringRef reg : {"kind", "source_endpoint", "sink_endpoint"}) {
        std::string registerName = prefix + reg.str();
        if (!seen.contains(configKey(target, registerName, source)))
          return llvm::createStringError(
              std::errc::invalid_argument,
              "config bitstream is missing route segment register %s for %s",
              registerName.c_str(), target.c_str());
      }
      if (!route.segments[segmentIndex].hardwareRef.empty()) {
        std::string registerName = prefix + "hardware_ref";
        if (!seen.contains(configKey(target, registerName, source)))
          return llvm::createStringError(
              std::errc::invalid_argument,
              "config bitstream is missing route segment register %s for %s",
              registerName.c_str(), target.c_str());
      }
    }
  }
  return llvm::Error::success();
}

namespace {

llvm::json::Object configJson(const ConfigEntry &entry) {
  return llvm::json::Object{
      {"target", entry.target},
      {"register", entry.registerName},
      {"value", entry.value},
      {"source", entry.source},
  };
}

llvm::json::Object resourcePressureJson(const ResourcePressureRecord &record) {
  return llvm::json::Object{
      {"resource_kind", record.resourceKind},
      {"operation", record.operation},
      {"required", static_cast<int64_t>(record.required)},
      {"available", static_cast<int64_t>(record.available)},
      {"placed", static_cast<int64_t>(record.placed)},
      {"missing", static_cast<int64_t>(record.missing)},
  };
}

} // namespace

} // namespace loom::pnr::detail

llvm::Error
loom::pnr::writeMappingCsv(llvm::StringRef outputPath,
                           llvm::ArrayRef<MappingSummary> summaries) {
  if (llvm::Error err = createParentDirectories(outputPath))
    return err;
  std::error_code ec;
  llvm::raw_fd_ostream out(outputPath, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return llvm::createStringError(ec, "could not open %s",
                                   outputPath.str().c_str());

  out << "workload,hardware,mapping_id,placed_records,routed_edges,"
         "unrouted_edges,unplaced_records,status,diagnostic\n";
  for (const MappingSummary &summary : summaries) {
    out << csvEscape(summary.workload) << ',' << csvEscape(summary.hardware)
        << ',' << csvEscape(summary.mappingId) << ','
        << summary.placements.size() << ',' << summary.routes.size() << ','
        << summary.unroutedEdges << ',' << summary.unplacedRecords << ','
        << csvEscape(summary.status) << ',' << csvEscape(summary.diagnostic)
        << '\n';
  }
  return llvm::Error::success();
}

llvm::Error loom::pnr::writeMappingJson(llvm::StringRef outputPath,
                                        const MappingSummary &summary) {
  if (llvm::Error err = createParentDirectories(outputPath))
    return err;

  llvm::json::Array placements;
  for (const PlacementRecord &placement : summary.placements)
    placements.push_back(placementJson(placement));

  llvm::json::Array routes;
  for (const RouteRecord &route : summary.routes)
    routes.push_back(routeJson(route));

  llvm::json::Array unroutedEdgeDetails;
  for (const UnroutedEdgeRecord &edge : summary.unroutedEdgeDetails)
    unroutedEdgeDetails.push_back(unroutedEdgeJson(edge));

  llvm::json::Array configEntries;
  for (const ConfigEntry &entry : summary.configEntries)
    configEntries.push_back(configJson(entry));

  llvm::json::Array resourcePressure;
  for (const ResourcePressureRecord &record : summary.resourcePressure)
    resourcePressure.push_back(resourcePressureJson(record));

  llvm::json::Object root{
      {"schema_version", "2.0"},
      {"kind", "pnr_mapping"},
      {"workload", summary.workload},
      {"hardware", summary.hardware},
      {"graph", summary.graph},
      {"mapping_id", summary.mappingId},
      {"config_id", summary.configId},
      {"config_fingerprint", summary.configFingerprint},
      {"component_config_view", summary.componentConfigView},
      {"component_config_fingerprint", summary.componentConfigFingerprint},
      {"status", summary.status},
      {"placed_records", static_cast<int64_t>(summary.placements.size())},
      {"routed_edges", static_cast<int64_t>(summary.routes.size())},
      {"unrouted_edges", static_cast<int64_t>(summary.unroutedEdges)},
      {"unplaced_records", static_cast<int64_t>(summary.unplacedRecords)},
      {"config_records", static_cast<int64_t>(summary.configEntries.size())},
      {"placements", std::move(placements)},
      {"routes", std::move(routes)},
      {"unrouted_edge_details", std::move(unroutedEdgeDetails)},
      {"config_bitstream", std::move(configEntries)},
  };
  if (summary.hardwareRootKind == "fabric.system") {
    root.try_emplace("hardware_root_kind", summary.hardwareRootKind);
    root.try_emplace("hardware_system", summary.hardwareSystem);
    root.try_emplace("selected_acc_core", summary.selectedAccCore);
    root.try_emplace("spatialcore_template", summary.spatialcoreTemplate);
  }
  if (!summary.resourcePressure.empty())
    root.try_emplace("resource_pressure", std::move(resourcePressure));

  if (!summary.diagnostic.empty()) {
    llvm::json::Array diagnostics;
    diagnostics.push_back(summary.diagnostic);
    root.try_emplace("diagnostics", std::move(diagnostics));
  }

  std::error_code ec;
  llvm::raw_fd_ostream out(outputPath, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return llvm::createStringError(ec, "could not open %s",
                                   outputPath.str().c_str());
  out << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << '\n';
  return llvm::Error::success();
}
