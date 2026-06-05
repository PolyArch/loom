#include "Simulator/CGRASimulator.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>
#include <system_error>

using namespace loom::sim;

namespace {

constexpr std::uint64_t kRouteLatencyPerEdge = 1;
constexpr std::uint64_t kMemoryLatencyPerAccess = 4;

struct ConfigEntries {
  llvm::StringMap<std::string> valuesByFullKey;
  llvm::StringSet<> writtenRegisters;
};

llvm::Error createParentDirectories(llvm::StringRef outputPath) {
  llvm::SmallString<256> parent(outputPath);
  llvm::sys::path::remove_filename(parent);
  if (parent.empty())
    return llvm::Error::success();
  if (std::error_code ec = llvm::sys::fs::create_directories(parent))
    return llvm::createStringError(ec, "could not create %s", parent.c_str());
  return llvm::Error::success();
}

llvm::Expected<llvm::json::Object> parseJsonObject(llvm::StringRef path) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (std::error_code ec = bufferOrErr.getError())
    return llvm::createStringError(ec, "could not read %s", path.str().c_str());
  auto parsedOrErr = llvm::json::parse((*bufferOrErr)->getBuffer());
  if (!parsedOrErr)
    return parsedOrErr.takeError();
  const llvm::json::Object *object = parsedOrErr->getAsObject();
  if (!object)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "%s is not a JSON object",
                                   path.str().c_str());
  return *object;
}

llvm::Expected<std::uint64_t> requireNonNegativeInteger(
    const llvm::json::Object &object, llvm::StringRef key,
    llvm::StringRef path) {
  std::optional<int64_t> value = object.getInteger(key);
  if (!value || *value < 0)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "%s lacks non-negative integer field %s",
                                   path.str().c_str(), key.str().c_str());
  return static_cast<std::uint64_t>(*value);
}

llvm::Expected<std::string> requireString(const llvm::json::Object &object,
                                          llvm::StringRef key,
                                          llvm::StringRef path) {
  std::optional<llvm::StringRef> value = object.getString(key);
  if (!value || value->empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "%s lacks string field %s",
                                   path.str().c_str(), key.str().c_str());
  return value->str();
}

llvm::Expected<std::string> requireObjectString(
    const llvm::json::Object &object, llvm::StringRef key,
    llvm::StringRef diagnosticContext) {
  std::optional<llvm::StringRef> value = object.getString(key);
  if (!value || value->empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "%s lacks string field %s",
                                   diagnosticContext.str().c_str(),
                                   key.str().c_str());
  return value->str();
}

llvm::Error requireKindAndPass(const llvm::json::Object &object,
                               llvm::StringRef expectedKind,
                               llvm::StringRef path) {
  std::optional<llvm::StringRef> kind = object.getString("kind");
  if (!kind || *kind != expectedKind)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "%s has wrong kind", path.str().c_str());
  std::optional<llvm::StringRef> status = object.getString("status");
  if (!status || *status != "pass")
    return llvm::createStringError(std::errc::invalid_argument,
                                   "%s is not a pass report",
                                   path.str().c_str());
  return llvm::Error::success();
}

bool isMemPlacement(llvm::StringRef resourceKind) {
  return resourceKind == "fabric.mem.load" || resourceKind == "fabric.mem.store";
}

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

llvm::Error expectConfig(const ConfigEntries &entries, llvm::StringRef target,
                         llvm::StringRef registerName, llvm::StringRef source,
                         llvm::StringRef expectedValue) {
  std::string key = configKey(target, registerName, source);
  auto it = entries.valuesByFullKey.find(key);
  if (it == entries.valuesByFullKey.end())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "mapping config bitstream is missing %s for %s",
        registerName.str().c_str(), target.str().c_str());
  if (it->second != expectedValue)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "mapping config bitstream value for %s on %s is %s, expected %s",
        registerName.str().c_str(), target.str().c_str(), it->second.c_str(),
        expectedValue.str().c_str());
  return llvm::Error::success();
}

llvm::Error collectConfigEntries(const llvm::json::Object &mapping,
                                 CGRASimReport &report,
                                 ConfigEntries &entries) {
  const llvm::json::Array *configArray = mapping.getArray("config_bitstream");
  if (!configArray)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "mapping artifact lacks config_bitstream");
  report.configRecords = configArray->size();
  for (const llvm::json::Value &value : *configArray) {
    const llvm::json::Object *entry = value.getAsObject();
    if (!entry)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "config bitstream entry is not an object");
    auto targetOrErr =
        requireObjectString(*entry, "target", "config bitstream entry");
    if (!targetOrErr)
      return targetOrErr.takeError();
    auto registerOrErr =
        requireObjectString(*entry, "register", "config bitstream entry");
    if (!registerOrErr)
      return registerOrErr.takeError();
    auto sourceOrErr =
        requireObjectString(*entry, "source", "config bitstream entry");
    if (!sourceOrErr)
      return sourceOrErr.takeError();
    auto valueOrErr =
        requireObjectString(*entry, "value", "config bitstream entry");
    if (!valueOrErr)
      return valueOrErr.takeError();

    std::string regKey = registerKey(*targetOrErr, *registerOrErr);
    if (!entries.writtenRegisters.insert(regKey).second)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "mapping config bitstream writes register %s on %s more than once",
          registerOrErr->c_str(), targetOrErr->c_str());
    std::string key = configKey(*targetOrErr, *registerOrErr, *sourceOrErr);
    if (!entries.valuesByFullKey.try_emplace(key, *valueOrErr).second)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "mapping config bitstream contains duplicate assignment for %s",
          targetOrErr->c_str());
  }
  return llvm::Error::success();
}

llvm::Error collectPlacementStats(const llvm::json::Object &mapping,
                                  CGRASimReport &report) {
  const llvm::json::Array *placements = mapping.getArray("placements");
  if (!placements)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "mapping artifact lacks placements");
  report.placedRecords = placements->size();
  std::uint64_t memPlacements = 0;
  for (const llvm::json::Value &value : *placements) {
    const llvm::json::Object *placement = value.getAsObject();
    if (!placement)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "mapping placement is not an object");
    std::optional<llvm::StringRef> schedule = placement->getString("schedule");
    if (schedule && *schedule == "temporal")
      ++report.temporalPlacements;
    else
      ++report.spatialPlacements;

    std::optional<llvm::StringRef> resourceKind =
        placement->getString("resource_kind");
    if (resourceKind && isMemPlacement(*resourceKind))
      ++memPlacements;
  }
  report.memoryLatencyCycles = memPlacements * kMemoryLatencyPerAccess;
  report.temporalPenaltyCycles =
      report.temporalPlacements == 0
          ? 0
          : report.temporalPlacements * (1 + report.routedEdges);
  return llvm::Error::success();
}

llvm::Error validateConfigCoverage(const llvm::json::Object &mapping,
                                   const CGRASimReport &report,
                                   const ConfigEntries &configEntries) {
  const llvm::json::Array *placements = mapping.getArray("placements");
  if (!placements)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "mapping artifact lacks placements");
  for (const llvm::json::Value &value : *placements) {
    const llvm::json::Object *placement = value.getAsObject();
    if (!placement)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "mapping placement is not an object");
    auto softwareOrErr =
        requireObjectString(*placement, "software", "mapping placement");
    if (!softwareOrErr)
      return softwareOrErr.takeError();
    auto hardwareOrErr =
        requireObjectString(*placement, "hardware", "mapping placement");
    if (!hardwareOrErr)
      return hardwareOrErr.takeError();
    auto operationOrErr =
        requireObjectString(*placement, "operation", "mapping placement");
    if (!operationOrErr)
      return operationOrErr.takeError();
    auto resourceKindOrErr =
        requireObjectString(*placement, "resource_kind", "mapping placement");
    if (!resourceKindOrErr)
      return resourceKindOrErr.takeError();
    auto scheduleOrErr =
        requireObjectString(*placement, "schedule", "mapping placement");
    if (!scheduleOrErr)
      return scheduleOrErr.takeError();
    std::string source = "placement:" + *softwareOrErr;
    if (llvm::Error err = expectConfig(configEntries, *hardwareOrErr,
                                       "software_id", source, *softwareOrErr))
      return err;
    if (llvm::Error err = expectConfig(configEntries, *hardwareOrErr,
                                       "operation", source, *operationOrErr))
      return err;
    if (llvm::Error err =
            expectConfig(configEntries, *hardwareOrErr, "resource_kind",
                         source, *resourceKindOrErr))
      return err;
    if (llvm::Error err = expectConfig(configEntries, *hardwareOrErr,
                                       "schedule", source, *scheduleOrErr))
      return err;
  }

  const llvm::json::Array *routes = mapping.getArray("routes");
  if (!routes)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "mapping artifact lacks routes");
  for (std::size_t i = 0; i < routes->size(); ++i) {
    const llvm::json::Object *route = (*routes)[i].getAsObject();
    if (!route)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "mapping route is not an object");
    auto fromOrErr = requireObjectString(*route, "from", "mapping route");
    if (!fromOrErr)
      return fromOrErr.takeError();
    auto toOrErr = requireObjectString(*route, "to", "mapping route");
    if (!toOrErr)
      return toOrErr.takeError();
    std::string source = "route:" + *fromOrErr + "->" + *toOrErr;
    std::string target =
        report.mappingId + "::route#" + std::to_string(i);
    if (llvm::Error err =
            expectConfig(configEntries, target, "from_software_id", source,
                         *fromOrErr))
      return err;
    if (llvm::Error err =
            expectConfig(configEntries, target, "to_software_id", source,
                         *toOrErr))
      return err;
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<CGRASimReport>
loom::sim::runCGRASimulation(const CGRASimOptions &options) {
  auto dfgOrErr = parseJsonObject(options.dfgReportPath);
  if (!dfgOrErr)
    return dfgOrErr.takeError();
  auto mappingOrErr = parseJsonObject(options.mappingArtifactPath);
  if (!mappingOrErr)
    return mappingOrErr.takeError();

  if (llvm::Error err =
          requireKindAndPass(*dfgOrErr, "dfg_sim_report", options.dfgReportPath))
    return std::move(err);
  if (llvm::Error err = requireKindAndPass(*mappingOrErr, "pnr_mapping",
                                           options.mappingArtifactPath))
    return std::move(err);

  CGRASimReport report;
  auto workloadOrErr =
      requireString(*dfgOrErr, "workload", options.dfgReportPath);
  if (!workloadOrErr)
    return workloadOrErr.takeError();
  report.workload = *workloadOrErr;

  auto mappingWorkloadOrErr =
      requireString(*mappingOrErr, "workload", options.mappingArtifactPath);
  if (!mappingWorkloadOrErr)
    return mappingWorkloadOrErr.takeError();
  if (*mappingWorkloadOrErr != report.workload)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "DFG report workload %s does not match mapping workload %s",
        report.workload.c_str(), mappingWorkloadOrErr->c_str());

  auto hardwareOrErr =
      requireString(*mappingOrErr, "hardware", options.mappingArtifactPath);
  if (!hardwareOrErr)
    return hardwareOrErr.takeError();
  report.hardware = *hardwareOrErr;
  auto mappingIdOrErr =
      requireString(*mappingOrErr, "mapping_id", options.mappingArtifactPath);
  if (!mappingIdOrErr)
    return mappingIdOrErr.takeError();
  report.mappingId = *mappingIdOrErr;

  auto dfgCyclesOrErr =
      requireNonNegativeInteger(*dfgOrErr, "optimistic_cycles",
                                options.dfgReportPath);
  if (!dfgCyclesOrErr)
    return dfgCyclesOrErr.takeError();
  report.dfgCycles = *dfgCyclesOrErr;
  auto routedEdgesOrErr =
      requireNonNegativeInteger(*mappingOrErr, "routed_edges",
                                options.mappingArtifactPath);
  if (!routedEdgesOrErr)
    return routedEdgesOrErr.takeError();
  report.routedEdges = *routedEdgesOrErr;
  report.routeLatencyCycles = report.routedEdges * kRouteLatencyPerEdge;

  ConfigEntries configEntries;
  if (llvm::Error err =
          collectConfigEntries(*mappingOrErr, report, configEntries))
    return std::move(err);
  if (llvm::Error err =
          validateConfigCoverage(*mappingOrErr, report, configEntries))
    return std::move(err);

  if (llvm::Error err = collectPlacementStats(*mappingOrErr, report))
    return std::move(err);

  report.hardwareAwareCycles = report.dfgCycles + report.routeLatencyCycles +
                               report.memoryLatencyCycles +
                               report.temporalPenaltyCycles;
  report.status = "pass";
  report.diagnostic =
      "CGRA-sim first-fidelity model: DFG cycles plus mapping route, "
      "memory-tile, and temporal reuse penalties";
  return report;
}

llvm::Error loom::sim::writeCGRASimReportJson(llvm::StringRef outputPath,
                                              const CGRASimReport &report) {
  if (llvm::Error err = createParentDirectories(outputPath))
    return err;

  llvm::json::Object root{
      {"schema_version", 1},
      {"kind", "cgra_sim_report"},
      {"workload", report.workload},
      {"hardware", report.hardware},
      {"mapping_id", report.mappingId},
      {"status", report.status},
      {"metric_definition", "dfg_plus_mapping_latency"},
      {"dfg_cycles", static_cast<int64_t>(report.dfgCycles)},
      {"route_latency_cycles",
       static_cast<int64_t>(report.routeLatencyCycles)},
      {"memory_latency_cycles",
       static_cast<int64_t>(report.memoryLatencyCycles)},
      {"temporal_penalty_cycles",
       static_cast<int64_t>(report.temporalPenaltyCycles)},
      {"hardware_aware_cycles",
       static_cast<int64_t>(report.hardwareAwareCycles)},
      {"placed_records", static_cast<int64_t>(report.placedRecords)},
      {"routed_edges", static_cast<int64_t>(report.routedEdges)},
      {"config_records", static_cast<int64_t>(report.configRecords)},
      {"spatial_placements", static_cast<int64_t>(report.spatialPlacements)},
      {"temporal_placements", static_cast<int64_t>(report.temporalPlacements)},
  };
  if (!report.diagnostic.empty()) {
    llvm::json::Array diagnostics;
    diagnostics.push_back(report.diagnostic);
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
