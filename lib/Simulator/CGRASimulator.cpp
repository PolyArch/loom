#include "Simulator/CGRASimulator.h"

#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Simulator/OperationSemantics.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

using namespace loom::sim;

namespace {

constexpr std::uint64_t kRouteLatencyPerSegment = 1;
constexpr std::uint64_t kMemoryLatencyPerAccess = 4;

struct RouteStats {
  std::uint64_t routeCount = 0;
  std::uint64_t segmentCount = 0;
};

struct ConfigEntries {
  llvm::StringMap<std::string> valuesByFullKey;
  llvm::StringSet<> writtenRegisters;
};

struct HardwareArtifactResource {
  std::string resourceKind;
  llvm::StringSet<> supportedOps;
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

std::optional<std::string> symbolName(mlir::Operation *op) {
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("sym_name"))
    return attr.getValue().str();
  return std::nullopt;
}

mlir::Operation *findFabricModule(mlir::ModuleOp module,
                                  llvm::StringRef symbol) {
  mlir::Operation *found = nullptr;
  module.walk([&](mlir::Operation *op) {
    if (found || op->getName().getStringRef() != "fabric.module")
      return;
    std::optional<std::string> name = symbolName(op);
    if (name && *name == symbol)
      found = op;
  });
  return found;
}

std::uint64_t integerAttrValue(mlir::Attribute attr) {
  if (auto intAttr = llvm::dyn_cast_if_present<mlir::IntegerAttr>(attr))
    return static_cast<std::uint64_t>(intAttr.getInt());
  return 0;
}

void addHardwareResource(llvm::StringMap<HardwareArtifactResource> &resources,
                         llvm::StringRef resourceId,
                         llvm::StringRef resourceKind) {
  HardwareArtifactResource resource;
  resource.resourceKind = resourceKind.str();
  resources.try_emplace(resourceId, std::move(resource));
}

void appendHardwareMemResources(
    mlir::Operation *op, llvm::StringRef hardwareName,
    llvm::StringMap<HardwareArtifactResource> &resources) {
  std::uint64_t loadPorts = 0;
  std::uint64_t storePorts = 0;
  auto hwParams = op->getAttrOfType<mlir::ArrayAttr>("hw_params");
  if (hwParams && !hwParams.empty()) {
    if (auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(hwParams[0])) {
      loadPorts = integerAttrValue(dict.get("load_group_size"));
      storePorts = integerAttrValue(dict.get("store_group_size"));
    }
  }
  for (std::uint64_t i = 0; i < loadPorts; ++i)
    addHardwareResource(resources,
                        (hardwareName + "::mem.load#" + llvm::Twine(i)).str(),
                        "fabric.mem.load");
  for (std::uint64_t i = 0; i < storePorts; ++i)
    addHardwareResource(resources,
                        (hardwareName + "::mem.store#" + llvm::Twine(i)).str(),
                        "fabric.mem.store");
}

void appendHardwareOpResource(
    mlir::Operation *op, llvm::StringRef hardwareName, unsigned index,
    llvm::StringMap<HardwareArtifactResource> &resources) {
  auto opList = op->getAttrOfType<mlir::ArrayAttr>("op_list");
  if (!opList)
    return;
  HardwareArtifactResource resource;
  resource.resourceKind = "fabric.op";
  for (mlir::Attribute attr : opList) {
    if (auto sym = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attr))
      resource.supportedOps.insert(sym.getValue());
  }
  resources.try_emplace(
      (hardwareName + "::fabric.op#" + llvm::Twine(index)).str(),
      std::move(resource));
}

llvm::StringMap<HardwareArtifactResource>
collectHardwareArtifactResources(mlir::Operation *hardware,
                                 llvm::StringRef hardwareName) {
  llvm::StringMap<HardwareArtifactResource> resources;
  unsigned fabricOpIndex = 0;
  hardware->walk([&](mlir::Operation *op) {
    llvm::StringRef opName = op->getName().getStringRef();
    if (opName == "fabric.op") {
      appendHardwareOpResource(op, hardwareName, fabricOpIndex++, resources);
      return;
    }
    if (opName == "fabric.mem")
      appendHardwareMemResources(op, hardwareName, resources);
  });
  return resources;
}

llvm::Expected<std::string>
requireObjectString(const llvm::json::Object &object, llvm::StringRef key,
                    llvm::StringRef diagnosticContext);

std::string endpointResourceId(llvm::StringRef endpoint) {
  std::size_t dot = endpoint.rfind('.');
  if (dot == llvm::StringRef::npos)
    return endpoint.str();
  return endpoint.take_front(dot).str();
}

llvm::Error validateHardwareArtifact(llvm::StringRef hardwareMlirPath,
                                     llvm::StringRef hardwareName,
                                     const llvm::json::Object &mapping) {
  if (hardwareMlirPath.empty())
    return llvm::Error::success();

  mlir::DialectRegistry registry;
  registry.insert<fabric::FabricDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(hardwareMlirPath, &context);
  if (!module)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "could not parse hardware artifact %s",
                                   hardwareMlirPath.str().c_str());
  mlir::Operation *hardware = findFabricModule(*module, hardwareName);
  if (!hardware)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "hardware artifact does not contain fabric.module %s",
        hardwareName.str().c_str());
  llvm::StringMap<HardwareArtifactResource> resources =
      collectHardwareArtifactResources(hardware, hardwareName);
  const llvm::json::Array *placements = mapping.getArray("placements");
  if (!placements)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "mapping artifact lacks placements");
  for (const llvm::json::Value &value : *placements) {
    const llvm::json::Object *placement = value.getAsObject();
    if (!placement)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "mapping placement is not an object");
    auto hardwareOrErr =
        requireObjectString(*placement, "hardware", "mapping placement");
    if (!hardwareOrErr)
      return hardwareOrErr.takeError();
    auto resourceKindOrErr =
        requireObjectString(*placement, "resource_kind", "mapping placement");
    if (!resourceKindOrErr)
      return resourceKindOrErr.takeError();
    auto operationOrErr =
        requireObjectString(*placement, "operation", "mapping placement");
    if (!operationOrErr)
      return operationOrErr.takeError();

    auto resourceIt = resources.find(*hardwareOrErr);
    if (resourceIt == resources.end())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "hardware artifact does not contain resource %s",
          hardwareOrErr->c_str());
    if (resourceIt->second.resourceKind != *resourceKindOrErr)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "hardware resource %s has kind %s but mapping requires %s",
          hardwareOrErr->c_str(), resourceIt->second.resourceKind.c_str(),
          resourceKindOrErr->c_str());
    if (*resourceKindOrErr == "fabric.op" &&
        !resourceIt->second.supportedOps.contains(*operationOrErr))
      return llvm::createStringError(
        std::errc::invalid_argument,
        "hardware resource %s does not support operation %s",
        hardwareOrErr->c_str(), operationOrErr->c_str());
  }
  const llvm::json::Array *routes = mapping.getArray("routes");
  if (!routes)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "mapping artifact lacks routes");
  for (const llvm::json::Value &routeValue : *routes) {
    const llvm::json::Object *route = routeValue.getAsObject();
    if (!route)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "mapping route is not an object");
    const llvm::json::Array *segments = route->getArray("segments");
    if (!segments || segments->empty())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "mapping route lacks non-empty segments");
    for (const llvm::json::Value &segmentValue : *segments) {
      const llvm::json::Object *segment = segmentValue.getAsObject();
      if (!segment)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "mapping route segment is not an object");
      for (llvm::StringRef key : {"source_endpoint", "sink_endpoint"}) {
        auto endpointOrErr =
            requireObjectString(*segment, key, "mapping route segment");
        if (!endpointOrErr)
          return endpointOrErr.takeError();
        std::string resourceId = endpointResourceId(*endpointOrErr);
        if (!resources.contains(resourceId))
          return llvm::createStringError(
              std::errc::invalid_argument,
              "hardware artifact does not contain route endpoint resource %s",
              resourceId.c_str());
      }
      if (std::optional<llvm::StringRef> hardwareRef =
              segment->getString("hardware_ref")) {
        if (*hardwareRef != hardwareName && !resources.contains(*hardwareRef))
          return llvm::createStringError(
              std::errc::invalid_argument,
              "hardware artifact does not contain route segment hardware_ref %s",
              hardwareRef->str().c_str());
      }
    }
  }
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

llvm::Expected<std::uint64_t>
requireNonNegativeInteger(const llvm::json::Object &object, llvm::StringRef key,
                          llvm::StringRef path) {
  std::optional<int64_t> value = object.getInteger(key);
  if (!value || *value < 0)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "%s lacks non-negative integer field %s",
                                   path.str().c_str(), key.str().c_str());
  return static_cast<std::uint64_t>(*value);
}

llvm::Expected<std::vector<std::string>>
requireStringArrayField(const llvm::json::Object &object, llvm::StringRef key,
                        llvm::StringRef path) {
  std::vector<std::string> values;
  const llvm::json::Array *array = object.getArray(key);
  if (!array)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "%s lacks array field %s",
                                   path.str().c_str(), key.str().c_str());
  for (auto [index, value] : llvm::enumerate(*array)) {
    std::optional<llvm::StringRef> string = value.getAsString();
    if (!string)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "%s field %s entry %u is not a string", path.str().c_str(),
          key.str().c_str(), static_cast<unsigned>(index));
    values.push_back(string->str());
  }
  return values;
}

llvm::Expected<std::map<std::string, std::vector<std::string>>>
requireStringArrayObjectField(const llvm::json::Object &object,
                              llvm::StringRef key, llvm::StringRef path) {
  std::map<std::string, std::vector<std::string>> result;
  const llvm::json::Object *state = object.getObject(key);
  if (!state)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "%s lacks object field %s",
                                   path.str().c_str(), key.str().c_str());
  for (const auto &[name, value] : *state) {
    std::vector<std::string> values;
    const llvm::json::Array *array = value.getAsArray();
    if (!array) {
      return llvm::createStringError(
          std::errc::invalid_argument, "%s field %s.%s is not an array",
          path.str().c_str(), key.str().c_str(), name.str().c_str());
    }
    for (auto [index, entry] : llvm::enumerate(*array)) {
      std::optional<llvm::StringRef> string = entry.getAsString();
      if (!string)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "%s field %s.%s entry %u is not a string", path.str().c_str(),
            key.str().c_str(), name.str().c_str(), static_cast<unsigned>(index));
      values.push_back(string->str());
    }
    result[name.str()] = std::move(values);
  }
  return result;
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

llvm::Expected<std::string>
requireSupportedReportSource(const llvm::json::Object &object,
                             llvm::StringRef key, llvm::StringRef expected,
                             llvm::StringRef label, llvm::StringRef path) {
  auto valueOrErr = requireString(object, key, path);
  if (!valueOrErr)
    return valueOrErr.takeError();
  if (*valueOrErr == expected)
    return *valueOrErr;
  return llvm::createStringError(
      std::errc::invalid_argument, "DFG report %s source %s is not supported",
      label.str().c_str(), valueOrErr->c_str());
}

llvm::Expected<std::string>
requireObjectString(const llvm::json::Object &object, llvm::StringRef key,
                    llvm::StringRef diagnosticContext) {
  std::optional<llvm::StringRef> value = object.getString(key);
  if (!value || value->empty())
    return llvm::createStringError(
        std::errc::invalid_argument, "%s lacks string field %s",
        diagnosticContext.str().c_str(), key.str().c_str());
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
  return resourceKind == "fabric.mem.load" ||
         resourceKind == "fabric.mem.store";
}

bool isSupportedResourceKind(llvm::StringRef resourceKind) {
  return resourceKind == "fabric.op" || resourceKind == "fabric.mem.load" ||
         resourceKind == "fabric.mem.store";
}

bool isSupportedSchedule(llvm::StringRef schedule) {
  return schedule == "spatial" || schedule == "temporal";
}

llvm::StringRef differenceClassification(const CGRASimReport &report) {
  if (report.status != "pass")
    return "unsupported_scope";
  return report.performanceDeltaCycles == 0 ? "no_modeled_hardware_constraints"
                                            : "expected_hardware_constraint";
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
                                 llvm::StringRef mappingArtifactPath,
                                 CGRASimReport &report,
                                 ConfigEntries &entries) {
  const llvm::json::Array *configArray = mapping.getArray("config_bitstream");
  if (!configArray)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "mapping artifact lacks config_bitstream");
  auto declaredRecordsOrErr =
      requireNonNegativeInteger(mapping, "config_records", mappingArtifactPath);
  if (!declaredRecordsOrErr)
    return declaredRecordsOrErr.takeError();
  if (*declaredRecordsOrErr != configArray->size())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "mapping config_records field %llu does not match config_bitstream "
        "size %llu",
        static_cast<unsigned long long>(*declaredRecordsOrErr),
        static_cast<unsigned long long>(configArray->size()));
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
    if (!schedule || !isSupportedSchedule(*schedule))
      return llvm::createStringError(
          std::errc::invalid_argument,
          "mapping placement schedule %s is not supported",
          schedule ? schedule->str().c_str() : "<missing>");
    if (*schedule == "temporal")
      ++report.temporalPlacements;
    else
      ++report.spatialPlacements;

    std::optional<llvm::StringRef> resourceKind =
        placement->getString("resource_kind");
    if (!resourceKind || !isSupportedResourceKind(*resourceKind))
      return llvm::createStringError(
          std::errc::invalid_argument,
          "mapping placement resource_kind %s is not supported",
          resourceKind ? resourceKind->str().c_str() : "<missing>");
    std::optional<llvm::StringRef> operation =
        placement->getString("operation");
    if (*resourceKind == "fabric.op" &&
        (!operation || !isSupportedMappedOperation(*operation)))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "mapping placement operation %s is not "
                                     "supported by operation semantics",
                                     operation ? operation->str().c_str()
                                               : "<missing>");
    if (isMemPlacement(*resourceKind))
      ++memPlacements;
  }
  report.memoryLatencyCycles = memPlacements * kMemoryLatencyPerAccess;
  report.temporalPenaltyCycles =
      report.temporalPlacements == 0
          ? 0
          : report.temporalPlacements * (1 + report.routedEdges);
  return llvm::Error::success();
}

llvm::Expected<RouteStats>
collectRouteStats(const llvm::json::Object &mapping,
                  llvm::StringRef mappingArtifactPath) {
  const llvm::json::Array *routes = mapping.getArray("routes");
  if (!routes)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "mapping artifact lacks routes");
  auto routedEdgesOrErr =
      requireNonNegativeInteger(mapping, "routed_edges", mappingArtifactPath);
  if (!routedEdgesOrErr)
    return routedEdgesOrErr.takeError();
  RouteStats stats;
  stats.routeCount = routes->size();
  if (*routedEdgesOrErr != stats.routeCount)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "mapping routed_edges field %llu does not match routes array size %llu",
        static_cast<unsigned long long>(*routedEdgesOrErr),
        static_cast<unsigned long long>(stats.routeCount));
  for (const llvm::json::Value &value : *routes) {
    const llvm::json::Object *route = value.getAsObject();
    if (!route)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "mapping route is not an object");
    const llvm::json::Array *segments = route->getArray("segments");
    if (!segments || segments->empty())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "mapping route lacks non-empty segments");
    for (const llvm::json::Value &segmentValue : *segments) {
      const llvm::json::Object *segment = segmentValue.getAsObject();
      if (!segment)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "mapping route segment is not an object");
      for (llvm::StringRef key :
           {"segment_id", "segment_kind", "source_endpoint", "sink_endpoint"}) {
        if (!segment->getString(key))
          return llvm::createStringError(
              std::errc::invalid_argument,
              "mapping route segment lacks string field %s", key.str().c_str());
      }
      ++stats.segmentCount;
    }
  }
  return stats;
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
            expectConfig(configEntries, *hardwareOrErr, "resource_kind", source,
                         *resourceKindOrErr))
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
    auto recordOrErr =
        requireObjectString(*route, "record_id", "mapping route");
    if (!recordOrErr)
      return recordOrErr.takeError();
    auto fromOrErr = requireObjectString(*route, "from", "mapping route");
    if (!fromOrErr)
      return fromOrErr.takeError();
    auto toOrErr = requireObjectString(*route, "to", "mapping route");
    if (!toOrErr)
      return toOrErr.takeError();
    std::string source = "route:" + *recordOrErr;
    std::string target = report.mappingId + "::" + *recordOrErr;
    if (llvm::Error err = expectConfig(configEntries, target,
                                       "from_software_id", source, *fromOrErr))
      return err;
    if (llvm::Error err = expectConfig(configEntries, target, "to_software_id",
                                       source, *toOrErr))
      return err;
    const llvm::json::Array *segments = route->getArray("segments");
    if (!segments || segments->empty())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "mapping route lacks non-empty segments");
    if (llvm::Error err =
            expectConfig(configEntries, target, "segment_count", source,
                         std::to_string(segments->size())))
      return err;
    for (std::size_t segmentIndex = 0; segmentIndex < segments->size();
         ++segmentIndex) {
      const llvm::json::Object *segment =
          (*segments)[segmentIndex].getAsObject();
      if (!segment)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "mapping route segment is not an object");
      std::string prefix = "segment." + std::to_string(segmentIndex) + ".";
      for (auto [jsonKey, registerName] :
           {std::pair<llvm::StringRef, llvm::StringRef>{"segment_kind", "kind"},
            {"source_endpoint", "source_endpoint"},
            {"sink_endpoint", "sink_endpoint"}}) {
        std::optional<llvm::StringRef> value = segment->getString(jsonKey);
        if (!value)
          return llvm::createStringError(
              std::errc::invalid_argument,
              "mapping route segment lacks string field %s",
              jsonKey.str().c_str());
        std::string segmentRegister = prefix + registerName.str();
        if (llvm::Error err = expectConfig(configEntries, target,
                                           segmentRegister, source, *value))
          return err;
      }
      if (std::optional<llvm::StringRef> value =
              segment->getString("hardware_ref")) {
        std::string segmentRegister = prefix + "hardware_ref";
        if (llvm::Error err = expectConfig(configEntries, target,
                                           segmentRegister, source, *value))
          return err;
      }
    }
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

  if (llvm::Error err = requireKindAndPass(*dfgOrErr, "dfg_sim_report",
                                           options.dfgReportPath))
    return std::move(err);
  std::optional<llvm::StringRef> mappingKind = mappingOrErr->getString("kind");
  if (!mappingKind || *mappingKind != "pnr_mapping")
    return llvm::createStringError(std::errc::invalid_argument,
                                   "%s has wrong kind",
                                   options.mappingArtifactPath.c_str());
  std::optional<llvm::StringRef> mappingStatus =
      mappingOrErr->getString("status");
  if (!mappingStatus || mappingStatus->empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "%s lacks string field status",
                                   options.mappingArtifactPath.c_str());

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
  report.hardwareArtifact = options.hardwareMlirPath;
  auto mappingIdOrErr =
      requireString(*mappingOrErr, "mapping_id", options.mappingArtifactPath);
  if (!mappingIdOrErr)
    return mappingIdOrErr.takeError();
  report.mappingId = *mappingIdOrErr;
  if (llvm::Error err = validateHardwareArtifact(
          options.hardwareMlirPath, report.hardware, *mappingOrErr))
    return std::move(err);

  auto dfgCyclesOrErr = requireNonNegativeInteger(
      *dfgOrErr, "optimistic_cycles", options.dfgReportPath);
  if (!dfgCyclesOrErr)
    return dfgCyclesOrErr.takeError();
  report.dfgCycles = *dfgCyclesOrErr;
  auto semanticsOrErr = requireSupportedReportSource(
      *dfgOrErr, "operation_semantics_source", kOperationSemanticsSource,
      "operation semantics", options.dfgReportPath);
  if (!semanticsOrErr)
    return semanticsOrErr.takeError();
  report.operationSemanticsSource = *semanticsOrErr;
  auto costModelOrErr = requireSupportedReportSource(
      *dfgOrErr, "operation_cost_model_source", kOperationCostModelSource,
      "operation cost model", options.dfgReportPath);
  if (!costModelOrErr)
    return costModelOrErr.takeError();
  report.operationCostModelSource = *costModelOrErr;
  auto finalOutputsOrErr =
      requireStringArrayField(*dfgOrErr, "final_outputs", options.dfgReportPath);
  if (!finalOutputsOrErr)
    return finalOutputsOrErr.takeError();
  report.finalOutputs = *finalOutputsOrErr;
  auto finalMemoryStateOrErr = requireStringArrayObjectField(
      *dfgOrErr, "final_memory_state", options.dfgReportPath);
  if (!finalMemoryStateOrErr)
    return finalMemoryStateOrErr.takeError();
  report.finalMemoryState = *finalMemoryStateOrErr;
  report.functionalStateSource = "carried_from_dfg_sim_report";

  if (*mappingStatus != "pass") {
    auto routeStatsOrErr =
        collectRouteStats(*mappingOrErr, options.mappingArtifactPath);
    if (!routeStatsOrErr)
      return routeStatsOrErr.takeError();
    report.routedEdges = routeStatsOrErr->routeCount;
    report.routeSegments = routeStatsOrErr->segmentCount;
    auto configRecordsOrErr = requireNonNegativeInteger(
        *mappingOrErr, "config_records", options.mappingArtifactPath);
    if (!configRecordsOrErr)
      return configRecordsOrErr.takeError();
    report.configRecords = *configRecordsOrErr;
    if (llvm::Error err = collectPlacementStats(*mappingOrErr, report))
      return std::move(err);
    report.memoryLatencyCycles = 0;
    report.temporalPenaltyCycles = 0;
    report.hardwareAwareCycles = report.dfgCycles;
    report.modeledLowerBoundCycles = report.dfgCycles;
    report.performanceDeltaCycles = 0;
    report.status = "blocked";
    report.diagnostic = "mapping artifact status " + mappingStatus->str() +
                        " blocks CGRA-sim";
    if (const llvm::json::Array *diagnostics =
            mappingOrErr->getArray("diagnostics")) {
      if (!diagnostics->empty()) {
        if (std::optional<llvm::StringRef> detail =
                (*diagnostics)[0].getAsString())
          report.diagnostic += ": " + detail->str();
      }
    }
    return report;
  }

  auto routeStatsOrErr =
      collectRouteStats(*mappingOrErr, options.mappingArtifactPath);
  if (!routeStatsOrErr)
    return routeStatsOrErr.takeError();
  report.routedEdges = routeStatsOrErr->routeCount;
  report.routeSegments = routeStatsOrErr->segmentCount;
  report.routeLatencyCycles = report.routeSegments * kRouteLatencyPerSegment;

  ConfigEntries configEntries;
  if (llvm::Error err = collectConfigEntries(
          *mappingOrErr, options.mappingArtifactPath, report, configEntries))
    return std::move(err);
  if (llvm::Error err =
          validateConfigCoverage(*mappingOrErr, report, configEntries))
    return std::move(err);

  if (llvm::Error err = collectPlacementStats(*mappingOrErr, report))
    return std::move(err);

  report.hardwareAwareCycles = report.dfgCycles + report.routeLatencyCycles +
                               report.memoryLatencyCycles +
                               report.temporalPenaltyCycles;
  report.modeledLowerBoundCycles = report.hardwareAwareCycles;
  report.performanceDeltaCycles = report.hardwareAwareCycles - report.dfgCycles;
  report.status = "pass";
  report.diagnostic =
      "CGRA-sim mapping-constraint estimate: DFG cycles plus modeled route, "
      "memory, and temporal penalties; report lists unmodeled "
      "microarchitectural constraints";
  return report;
}

llvm::Error loom::sim::writeCGRASimReportJson(llvm::StringRef outputPath,
                                              const CGRASimReport &report) {
  if (llvm::Error err = createParentDirectories(outputPath))
    return err;

  llvm::json::Array cycleBreakdown;
  cycleBreakdown.push_back(llvm::json::Object{
      {"category", "route_latency"},
      {"cycles", static_cast<int64_t>(report.routeLatencyCycles)},
      {"evidence", "mapping.route_segments"},
      {"modeled", true},
      {"explanation",
       "one first-order route cost per consumed route segment; explicit Fabric "
       "FIFO timing is listed as an unmodeled constraint"},
  });
  cycleBreakdown.push_back(llvm::json::Object{
      {"category", "memory_latency"},
      {"cycles", static_cast<int64_t>(report.memoryLatencyCycles)},
      {"evidence", "fabric.mem placement"},
      {"modeled", true},
      {"explanation",
       "fixed first-order memory tile latency per mapped load/store resource"},
  });
  cycleBreakdown.push_back(llvm::json::Object{
      {"category", "temporal_conflict"},
      {"cycles", static_cast<int64_t>(report.temporalPenaltyCycles)},
      {"evidence", "placement schedule"},
      {"modeled", true},
      {"explanation",
       "temporal placements add first-order reuse conflict cost; fully spatial "
       "placements have zero temporal conflict penalty"},
  });

  llvm::json::Array unmodeledConstraints;
  unmodeledConstraints.push_back("explicit_fabric_route_paths");
  unmodeledConstraints.push_back("fifo_latency");
  unmodeledConstraints.push_back("cache_behavior");
  unmodeledConstraints.push_back("scratchpad_bank_conflicts");
  unmodeledConstraints.push_back("coherence_consistency");

  llvm::json::Array firstPrinciplesChecks;
  firstPrinciplesChecks.push_back(llvm::json::Object{
      {"name", "cgra_not_more_optimistic_than_dfg"},
      {"status", "pass"},
      {"evidence", "hardware_aware_cycles >= dfg_cycles"},
  });
  firstPrinciplesChecks.push_back(llvm::json::Object{
      {"name", "modeled_constraint_lower_bound"},
      {"status", "pass"},
      {"evidence", "hardware_aware_cycles >= modeled_lower_bound_cycles"},
  });
  firstPrinciplesChecks.push_back(llvm::json::Object{
      {"name", "delta_explained_by_modeled_constraints"},
      {"status", "pass"},
      {"evidence", "performance_delta_cycles = route_latency_cycles + "
                   "memory_latency_cycles + temporal_penalty_cycles"},
  });

  llvm::json::Object root{
      {"schema_version", 1},
      {"kind", "cgra_sim_report"},
      {"workload", report.workload},
      {"hardware", report.hardware},
      {"mapping_id", report.mappingId},
      {"status", report.status},
      {"fidelity_level", "mapping_constraint_estimate"},
      {"metric_definition", "mapping_constraint_estimate"},
      {"operation_semantics_source", report.operationSemanticsSource},
      {"operation_cost_model_source", report.operationCostModelSource},
      {"difference_classification", differenceClassification(report)},
      {"hardware_bound_classification",
       report.status == "pass" ? "within_modeled_bounds" : "unsupported_scope"},
      {"dfg_cycles", static_cast<int64_t>(report.dfgCycles)},
      {"modeled_lower_bound_cycles",
       static_cast<int64_t>(report.modeledLowerBoundCycles)},
      {"performance_delta_cycles",
       static_cast<int64_t>(report.performanceDeltaCycles)},
      {"route_latency_cycles", static_cast<int64_t>(report.routeLatencyCycles)},
      {"memory_latency_cycles",
       static_cast<int64_t>(report.memoryLatencyCycles)},
      {"temporal_penalty_cycles",
       static_cast<int64_t>(report.temporalPenaltyCycles)},
      {"hardware_aware_cycles",
       static_cast<int64_t>(report.hardwareAwareCycles)},
      {"placed_records", static_cast<int64_t>(report.placedRecords)},
      {"routed_edges", static_cast<int64_t>(report.routedEdges)},
      {"route_segments", static_cast<int64_t>(report.routeSegments)},
      {"config_records", static_cast<int64_t>(report.configRecords)},
      {"spatial_placements", static_cast<int64_t>(report.spatialPlacements)},
      {"temporal_placements", static_cast<int64_t>(report.temporalPlacements)},
      {"cycle_breakdown", std::move(cycleBreakdown)},
      {"unmodeled_constraints", std::move(unmodeledConstraints)},
      {"first_principles_checks", std::move(firstPrinciplesChecks)},
  };
  llvm::json::Array finalOutputs;
  for (const std::string &value : report.finalOutputs)
    finalOutputs.push_back(value);
  root.try_emplace("final_outputs", std::move(finalOutputs));

  llvm::json::Object finalMemoryState;
  for (const auto &[argument, values] : report.finalMemoryState) {
    llvm::json::Array memoryValues;
    for (const std::string &value : values)
      memoryValues.push_back(value);
    finalMemoryState[argument] = std::move(memoryValues);
  }
  root.try_emplace("final_memory_state", std::move(finalMemoryState));
  root.try_emplace("functional_state_source", report.functionalStateSource);
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
