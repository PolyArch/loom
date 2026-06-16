#include "PnR/Mapping.h"

#include "Common/ResolvedConfig.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <tuple>

using namespace loom::pnr;

namespace {

enum class ResourceKind {
  FabricOp,
  MemLoad,
  MemStore,
};

struct SoftwareNode {
  std::string id;
  std::string operation;
  ResourceKind resourceKind;
  mlir::Operation *op = nullptr;
};

struct HardwareResource {
  std::string id;
  ResourceKind kind;
  std::string schedule;
  mlir::Operation *op = nullptr;
  std::map<std::string, std::set<std::string>> hwParamOptions;
  std::map<std::string, std::string> swConfigs;
  llvm::StringSet<> supportedOps;
  bool used = false;
};

struct HardwareRouteSegment {
  std::string segmentKind;
  std::string sourceEndpoint;
  std::string sinkEndpoint;
  std::string hardwareRef;
};

struct HardwareTopology {
  std::map<std::string, llvm::SmallVector<HardwareRouteSegment, 2>>
      segmentsBySource;
};

struct EndpointKey {
  mlir::Operation *op = nullptr;
  unsigned index = 0;

  bool operator<(const EndpointKey &other) const {
    return std::tie(op, index) < std::tie(other.op, other.index);
  }
};

struct HardwareModel {
  llvm::SmallVector<HardwareResource> resources;
  HardwareTopology topology;
};

struct RouteCollection {
  llvm::SmallVector<RouteRecord, 0> routes;
  llvm::SmallVector<UnroutedEdgeRecord, 0> unroutedEdgeDetails;
  std::uint64_t unroutedEdges = 0;
};

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

std::string escapedIdentityPart(llvm::StringRef value) {
  std::string part;
  constexpr char hex[] = "0123456789ABCDEF";
  for (char ch : value) {
    unsigned char byte = static_cast<unsigned char>(ch);
    if (llvm::isAlnum(ch) || ch == '_') {
      part.push_back(ch);
      continue;
    }
    part.push_back('%');
    part.push_back(hex[(byte >> 4) & 0xF]);
    part.push_back(hex[byte & 0xF]);
  }
  return part;
}

std::string mappingId(llvm::StringRef workload, llvm::StringRef graph,
                      llvm::StringRef hardware) {
  std::string id = escapedIdentityPart(workload);
  id += "__";
  id += escapedIdentityPart(graph);
  id += "__";
  id += escapedIdentityPart(hardware);
  return id;
}

mlir::DialectRegistry makeRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, fabric::FabricDialect,
                  mlir::arith::ArithDialect, mlir::DLTIDialect,
                  mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                  mlir::scf::SCFDialect, mlir::ub::UBDialect>();
  return registry;
}

mlir::OwningOpRef<mlir::ModuleOp> parseModule(mlir::MLIRContext &context,
                                              llvm::StringRef path) {
  return mlir::parseSourceFile<mlir::ModuleOp>(path, &context);
}

std::optional<std::string> symbolName(mlir::Operation *op) {
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("sym_name"))
    return attr.getValue().str();
  return std::nullopt;
}

mlir::Operation *findSymbolOp(mlir::ModuleOp module, llvm::StringRef opName,
                              llvm::StringRef symbol) {
  mlir::Operation *found = nullptr;
  module.walk([&](mlir::Operation *op) {
    if (found || op->getName().getStringRef() != opName)
      return;
    std::optional<std::string> name = symbolName(op);
    if (name && *name == symbol)
      found = op;
  });
  return found;
}

bool isIgnoredOp(mlir::Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  return name == "dataflow.graph.return";
}

bool isAdapterOp(mlir::Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  return name == "builtin.unrealized_conversion_cast" ||
         name == "arith.index_cast";
}

bool isLlvmPointerType(mlir::Type type) {
  return mlir::isa<mlir::LLVM::LLVMPointerType>(type);
}

bool isPointerCarryOp(mlir::Operation *op) {
  if (op->getName().getStringRef() != "dataflow.carry" ||
      op->getNumResults() != 1)
    return false;
  return isLlvmPointerType(op->getResult(0).getType());
}

bool isGraphReturnOp(mlir::Operation *op) {
  return op->getName().getStringRef() == "dataflow.graph.return";
}

bool isPointerBookkeepingOp(mlir::Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  if (name == "llvm.getelementptr") {
    if (op->getNumResults() != 1 ||
        !isLlvmPointerType(op->getResult(0).getType()))
      return false;
    for (mlir::OpOperand &use : op->getResult(0).getUses()) {
      mlir::Operation *owner = use.getOwner();
      if (isPointerCarryOp(owner) || isGraphReturnOp(owner))
        continue;
      return false;
    }
    return true;
  }

  if (!isPointerCarryOp(op))
    return false;
  for (mlir::OpOperand &use : op->getResult(0).getUses()) {
    mlir::Operation *owner = use.getOwner();
    llvm::StringRef ownerName = owner->getName().getStringRef();
    if (ownerName == "llvm.getelementptr" || isGraphReturnOp(owner))
      continue;
    return false;
  }
  return true;
}

std::optional<ResourceKind> resourceKindForSoftwareOp(mlir::Operation *op) {
  std::string nameStorage;
  llvm::StringRef name = op->getName().getStringRef();
  if (auto intrinsic = mlir::dyn_cast<mlir::LLVM::CallIntrinsicOp>(op)) {
    nameStorage = intrinsic.getIntrin().str();
    name = nameStorage;
  }
  if (name == "dataflow.load")
    return ResourceKind::MemLoad;
  if (name == "dataflow.store")
    return ResourceKind::MemStore;
  if (fabric::isFabricOpSupported(name))
    return ResourceKind::FabricOp;
  return std::nullopt;
}

std::string softwareOperationName(mlir::Operation *op) {
  if (auto intrinsic = mlir::dyn_cast<mlir::LLVM::CallIntrinsicOp>(op))
    return intrinsic.getIntrin().str();
  return op->getName().getStringRef().str();
}

llvm::StringRef resourceKindName(ResourceKind kind) {
  switch (kind) {
  case ResourceKind::FabricOp:
    return "fabric.op";
  case ResourceKind::MemLoad:
    return "fabric.mem.load";
  case ResourceKind::MemStore:
    return "fabric.mem.store";
  }
  llvm_unreachable("unknown resource kind");
}

llvm::Expected<llvm::SmallVector<SoftwareNode>>
collectSoftwareNodes(mlir::Operation *graph) {
  llvm::SmallVector<SoftwareNode> nodes;
  llvm::StringMap<unsigned> counts;
  for (mlir::Operation &op : graph->getRegion(0).front()) {
    if (!isGraphReturnOp(&op))
      continue;
    for (mlir::Value value : op.getOperands()) {
      if (isLlvmPointerType(value.getType()))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "graph returns unsupported pointer value for PnR mapping");
    }
  }
  for (mlir::Operation &op : graph->getRegion(0).front()) {
    if (isIgnoredOp(&op) || isAdapterOp(&op) || isPointerBookkeepingOp(&op))
      continue;
    std::optional<ResourceKind> kind = resourceKindForSoftwareOp(&op);
    if (!kind) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "graph contains unsupported operation for PnR mapping: %s",
          op.getName().getStringRef().str().c_str());
    }
    std::string opName = softwareOperationName(&op);
    unsigned index = counts[opName]++;
    nodes.push_back(
        SoftwareNode{opName + "#" + std::to_string(index), opName, *kind, &op});
  }
  return nodes;
}

std::uint64_t integerAttrValue(mlir::Attribute attr) {
  if (auto intAttr = llvm::dyn_cast_if_present<mlir::IntegerAttr>(attr))
    return static_cast<std::uint64_t>(intAttr.getInt());
  return 0;
}

std::string configValue(mlir::Attribute attr) {
  if (auto stringAttr = llvm::dyn_cast_if_present<mlir::StringAttr>(attr))
    return stringAttr.getValue().str();
  std::string text;
  llvm::raw_string_ostream os(text);
  attr.print(os);
  return text;
}

std::string apintToHexString(const llvm::APInt &value) {
  llvm::SmallString<32> hex;
  value.toString(hex, /*Radix=*/16, /*Signed=*/false,
                 /*formatAsCLiteral=*/false);
  for (char &ch : hex) {
    if (ch >= 'A' && ch <= 'F')
      ch = static_cast<char>(ch - 'A' + 'a');
  }
  std::string out = "0x";
  out += hex.c_str();
  return out;
}

std::optional<std::string> encodeConstHex(mlir::Attribute attr) {
  if (auto integer = llvm::dyn_cast_if_present<mlir::IntegerAttr>(attr))
    return apintToHexString(integer.getValue());
  if (auto fp = llvm::dyn_cast_if_present<mlir::FloatAttr>(attr))
    return apintToHexString(fp.getValue().bitcastToAPInt());
  if (auto stringAttr = llvm::dyn_cast_if_present<mlir::StringAttr>(attr)) {
    llvm::StringRef value = stringAttr.getValue();
    if (value.starts_with("0x") || value.starts_with("0X"))
      return value.str();
    return ("0x" + value).str();
  }
  return std::nullopt;
}

std::optional<std::string> canonicalHexValue(llvm::StringRef value) {
  if (!(value.starts_with("0x") || value.starts_with("0X")))
    return std::nullopt;
  llvm::StringRef digits = value.drop_front(2);
  if (digits.empty())
    return std::nullopt;
  std::string lowered;
  lowered.reserve(digits.size());
  for (char ch : digits) {
    if (!llvm::isHexDigit(ch))
      return std::nullopt;
    if (ch >= 'A' && ch <= 'F')
      ch = static_cast<char>(ch - 'A' + 'a');
    lowered.push_back(ch);
  }
  std::size_t firstNonZero = lowered.find_first_not_of('0');
  if (firstNonZero == std::string::npos)
    return std::string("0x0");
  return "0x" + lowered.substr(firstNonZero);
}

std::string scheduleName(fabric::Schedule schedule) {
  return fabric::stringifySchedule(schedule).str();
}

std::optional<std::string> predicateConfig(mlir::Operation *op) {
  if (auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(op))
    return mlir::arith::stringifyCmpIPredicate(cmp.getPredicate()).str();
  if (auto cmp = mlir::dyn_cast<mlir::arith::CmpFOp>(op))
    return mlir::arith::stringifyCmpFPredicate(cmp.getPredicate()).str();
  return std::nullopt;
}

std::optional<std::string> constantConfig(mlir::Operation *op) {
  if (op->getName().getStringRef() != "dataflow.constant")
    return std::nullopt;
  return encodeConstHex(op->getAttr("const_value"));
}

std::map<std::string, std::string>
softwareConfigsFor(const SoftwareNode &node) {
  std::map<std::string, std::string> configs;
  if (std::optional<std::string> predicate = predicateConfig(node.op))
    configs.try_emplace("predicate", *predicate);
  if (std::optional<std::string> constant = constantConfig(node.op))
    configs.try_emplace("const_hex_value", *constant);
  return configs;
}

std::string nearestSchedule(mlir::Operation *op) {
  for (mlir::Operation *cursor = op; cursor; cursor = cursor->getParentOp()) {
    if (auto attr = cursor->getAttrOfType<fabric::ScheduleAttr>("schedule"))
      return scheduleName(attr.getValue());
  }
  return "spatial";
}

void appendHwParamOptions(mlir::Operation *op, HardwareResource &resource) {
  auto hwParams = op->getAttrOfType<mlir::ArrayAttr>("hw_params");
  if (!hwParams)
    return;
  for (mlir::Attribute paramSet : hwParams) {
    auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(paramSet);
    if (!dict)
      continue;
    for (mlir::NamedAttribute namedAttr : dict) {
      std::set<std::string> &values =
          resource.hwParamOptions[namedAttr.getName().getValue().str()];
      if (auto array = mlir::dyn_cast<mlir::ArrayAttr>(namedAttr.getValue())) {
        for (mlir::Attribute value : array)
          values.insert(configValue(value));
        continue;
      }
      values.insert(configValue(namedAttr.getValue()));
    }
  }
}

void appendMemResources(mlir::Operation *op, llvm::StringRef hardwareName,
                        llvm::SmallVectorImpl<HardwareResource> &resources) {
  std::uint64_t loadPorts = 0;
  std::uint64_t storePorts = 0;
  auto hwParams = op->getAttrOfType<mlir::ArrayAttr>("hw_params");
  if (hwParams && !hwParams.empty()) {
    if (auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(hwParams[0])) {
      loadPorts = integerAttrValue(dict.get("load_group_size"));
      storePorts = integerAttrValue(dict.get("store_group_size"));
    }
  }
  std::string schedule = nearestSchedule(op);
  for (std::uint64_t i = 0; i < loadPorts; ++i) {
    resources.push_back(
        HardwareResource{(hardwareName + "::mem.load#" + llvm::Twine(i)).str(),
                         ResourceKind::MemLoad,
                         schedule,
                         op,
                         {},
                         {},
                         {},
                         false});
  }
  for (std::uint64_t i = 0; i < storePorts; ++i) {
    resources.push_back(
        HardwareResource{(hardwareName + "::mem.store#" + llvm::Twine(i)).str(),
                         ResourceKind::MemStore,
                         schedule,
                         op,
                         {},
                         {},
                         {},
                         false});
  }
}

void appendFabricOpResource(
    mlir::Operation *op, llvm::StringRef hardwareName, unsigned index,
    llvm::SmallVectorImpl<HardwareResource> &resources) {
  auto opList = op->getAttrOfType<mlir::ArrayAttr>("op_list");
  if (!opList)
    return;
  HardwareResource resource;
  resource.id = (hardwareName + "::fabric.op#" + llvm::Twine(index)).str();
  resource.kind = ResourceKind::FabricOp;
  resource.schedule = nearestSchedule(op);
  resource.op = op;
  appendHwParamOptions(op, resource);
  if (auto swConfigs = op->getAttrOfType<mlir::DictionaryAttr>("sw_configs")) {
    for (mlir::NamedAttribute namedAttr : swConfigs) {
      resource.swConfigs[namedAttr.getName().getValue().str()] =
          configValue(namedAttr.getValue());
    }
  }
  for (mlir::Attribute attr : opList) {
    if (auto sym = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attr))
      resource.supportedOps.insert(sym.getValue());
  }
  resources.push_back(std::move(resource));
}

std::string endpointFor(llvm::StringRef resourceId, llvm::StringRef endpoint,
                        unsigned index) {
  return (resourceId + "." + endpoint + llvm::Twine(index)).str();
}

bool isRouteResourceOp(llvm::StringRef opName) {
  return opName == "fabric.switch" || opName == "fabric.fifo" ||
         opName == "fabric.boundary";
}

std::string routeResourceKind(llvm::StringRef opName) {
  if (opName == "fabric.switch")
    return "fabric.switch";
  if (opName == "fabric.fifo")
    return "fabric.fifo";
  return "fabric.boundary";
}

std::string internalRouteSegmentKind(llvm::StringRef opName) {
  if (opName == "fabric.fifo")
    return "buffer";
  if (opName == "fabric.boundary")
    return "boundary_crossing";
  return "module_path";
}

llvm::SmallVector<std::string, 4> switchConnectivityRows(mlir::Operation *op) {
  llvm::SmallVector<std::string, 4> rows;
  auto hwParams = op->getAttrOfType<mlir::ArrayAttr>("hw_params");
  if (!hwParams || hwParams.empty())
    return rows;
  auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(hwParams[0]);
  if (!dict)
    return rows;
  auto table = llvm::dyn_cast_if_present<mlir::ArrayAttr>(
      dict.get("connectivity_table"));
  if (!table)
    return rows;
  for (mlir::Attribute attr : table) {
    auto row = llvm::dyn_cast<mlir::StringAttr>(attr);
    if (!row)
      return {};
    rows.push_back(row.getValue().str());
  }
  return rows;
}

bool switchConnectsInputToOutput(llvm::ArrayRef<std::string> rows,
                                 unsigned inputIndex, unsigned outputIndex,
                                 unsigned inputCount) {
  if (outputIndex >= rows.size())
    return false;
  llvm::StringRef row = rows[outputIndex];
  if (row.size() != inputCount || inputIndex >= inputCount)
    return false;
  return row[inputCount - 1 - inputIndex] == '1';
}

void addTopologySegment(HardwareTopology &topology,
                        HardwareRouteSegment segment) {
  topology.segmentsBySource[segment.sourceEndpoint].push_back(
      std::move(segment));
}

std::pair<std::uint64_t, std::uint64_t> memPortCounts(mlir::Operation *op) {
  std::uint64_t loadPorts = 0;
  std::uint64_t storePorts = 0;
  auto hwParams = op->getAttrOfType<mlir::ArrayAttr>("hw_params");
  if (hwParams && !hwParams.empty()) {
    if (auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(hwParams[0])) {
      loadPorts = integerAttrValue(dict.get("load_group_size"));
      storePorts = integerAttrValue(dict.get("store_group_size"));
    }
  }
  return {loadPorts, storePorts};
}

unsigned memResultPortBase(mlir::Operation *op) {
  if (op->getNumResults() == 0)
    return 0;
  if (mlir::isa<mlir::MemRefType>(op->getResult(0).getType()))
    return 1;
  return 0;
}

std::optional<std::string> forwardedBlockArgumentEndpoint(
    mlir::Value value,
    const std::map<EndpointKey, std::string> &resultEndpoints);

std::optional<std::string> sourceEndpointForValue(
    mlir::Value value,
    const std::map<EndpointKey, std::string> &resultEndpoints) {
  if (auto opResult = llvm::dyn_cast<mlir::OpResult>(value)) {
    auto it = resultEndpoints.find(
        EndpointKey{opResult.getOwner(), opResult.getResultNumber()});
    if (it != resultEndpoints.end())
      return it->second;
    return std::nullopt;
  }
  return forwardedBlockArgumentEndpoint(value, resultEndpoints);
}

std::optional<std::string> forwardedBlockArgumentEndpoint(
    mlir::Value value,
    const std::map<EndpointKey, std::string> &resultEndpoints) {
  auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(value);
  if (!blockArg)
    return std::nullopt;
  mlir::Operation *parent = blockArg.getOwner()->getParentOp();
  if (!parent)
    return std::nullopt;
  llvm::StringRef parentName = parent->getName().getStringRef();
  if (parentName != "fabric.fu" && parentName != "fabric.pe")
    return std::nullopt;
  unsigned index = blockArg.getArgNumber();
  if (index >= parent->getNumOperands())
    return std::nullopt;
  return sourceEndpointForValue(parent->getOperand(index), resultEndpoints);
}

void addGenericEndpointMaps(
    mlir::Operation *op, llvm::StringRef resourceId,
    std::map<EndpointKey, std::string> &operandEndpoints,
    std::map<EndpointKey, std::string> &resultEndpoints) {
  for (unsigned operandIndex = 0; operandIndex < op->getNumOperands();
       ++operandIndex)
    operandEndpoints.try_emplace(
        EndpointKey{op, operandIndex},
        endpointFor(resourceId, "operand", operandIndex));
  for (unsigned resultIndex = 0; resultIndex < op->getNumResults();
       ++resultIndex)
    resultEndpoints.try_emplace(EndpointKey{op, resultIndex},
                                endpointFor(resourceId, "result", resultIndex));
}

bool isFabricBoundaryOp(llvm::StringRef opName) {
  return opName == "fabric.fu" || opName == "fabric.pe";
}

void addMemEndpointMaps(mlir::Operation *op, llvm::StringRef hardwareName,
                        std::map<EndpointKey, std::string> &operandEndpoints,
                        std::map<EndpointKey, std::string> &resultEndpoints) {
  auto [loadPorts, storePorts] = memPortCounts(op);
  unsigned operandBase = 1;
  unsigned resultBase = memResultPortBase(op);
  for (std::uint64_t i = 0; i < loadPorts; ++i) {
    std::string resourceId =
        (hardwareName + "::mem.load#" + llvm::Twine(i)).str();
    operandEndpoints.try_emplace(EndpointKey{op, operandBase},
                                 endpointFor(resourceId, "operand", 0));
    operandEndpoints.try_emplace(EndpointKey{op, operandBase + 1},
                                 endpointFor(resourceId, "operand", 1));
    resultEndpoints.try_emplace(EndpointKey{op, resultBase},
                                endpointFor(resourceId, "result", 0));
    resultEndpoints.try_emplace(EndpointKey{op, resultBase + 1},
                                endpointFor(resourceId, "result", 1));
    operandBase += 2;
    resultBase += 2;
  }
  for (std::uint64_t i = 0; i < storePorts; ++i) {
    std::string resourceId =
        (hardwareName + "::mem.store#" + llvm::Twine(i)).str();
    operandEndpoints.try_emplace(EndpointKey{op, operandBase},
                                 endpointFor(resourceId, "operand", 0));
    operandEndpoints.try_emplace(EndpointKey{op, operandBase + 1},
                                 endpointFor(resourceId, "operand", 1));
    operandEndpoints.try_emplace(EndpointKey{op, operandBase + 2},
                                 endpointFor(resourceId, "operand", 2));
    resultEndpoints.try_emplace(EndpointKey{op, resultBase},
                                endpointFor(resourceId, "result", 0));
    operandBase += 3;
    resultBase += 1;
  }
}

void addYieldBoundarySegments(
    mlir::Operation *op,
    const llvm::DenseMap<mlir::Operation *, std::string> &fabricBoundaryIds,
    const std::map<EndpointKey, std::string> &resultEndpoints,
    HardwareTopology &topology) {
  if (op->getName().getStringRef() != "fabric.yield")
    return;
  mlir::Operation *parent = op->getParentOp();
  if (!parent)
    return;
  if (!isFabricBoundaryOp(parent->getName().getStringRef()))
    return;
  auto parentId = fabricBoundaryIds.find(parent);
  if (parentId == fabricBoundaryIds.end())
    return;
  for (unsigned operandIndex = 0; operandIndex < op->getNumOperands();
       ++operandIndex) {
    std::optional<std::string> sourceEndpoint =
        sourceEndpointForValue(op->getOperand(operandIndex), resultEndpoints);
    if (!sourceEndpoint)
      continue;
    auto sinkIt = resultEndpoints.find(EndpointKey{parent, operandIndex});
    if (sinkIt == resultEndpoints.end())
      continue;
    addTopologySegment(topology,
                       HardwareRouteSegment{"module_path", *sourceEndpoint,
                                            sinkIt->second, parentId->second});
  }
}

void addFuToPeBoundarySegments(
    mlir::Operation *op,
    const llvm::DenseMap<mlir::Operation *, std::string> &fabricBoundaryIds,
    const std::map<EndpointKey, std::string> &resultEndpoints,
    HardwareTopology &topology) {
  if (op->getName().getStringRef() != "fabric.fu")
    return;
  mlir::Operation *parent = op->getParentOp();
  if (!parent || parent->getName().getStringRef() != "fabric.pe")
    return;
  auto peId = fabricBoundaryIds.find(parent);
  if (peId == fabricBoundaryIds.end())
    return;
  unsigned resultCount = std::min(op->getNumResults(), parent->getNumResults());
  for (unsigned resultIndex = 0; resultIndex < resultCount; ++resultIndex) {
    auto sourceIt = resultEndpoints.find(EndpointKey{op, resultIndex});
    auto sinkIt = resultEndpoints.find(EndpointKey{parent, resultIndex});
    if (sourceIt == resultEndpoints.end() || sinkIt == resultEndpoints.end())
      continue;
    addTopologySegment(topology,
                       HardwareRouteSegment{"module_path", sourceIt->second,
                                            sinkIt->second, peId->second});
  }
}

std::optional<unsigned>
hardwareOperandIndexForSoftwareEndpoint(ResourceKind kind,
                                        unsigned softwareOperandIndex) {
  switch (kind) {
  case ResourceKind::FabricOp:
    return softwareOperandIndex;
  case ResourceKind::MemLoad:
    if (softwareOperandIndex == 1)
      return 0;
    if (softwareOperandIndex == 2)
      return 1;
    return std::nullopt;
  case ResourceKind::MemStore:
    if (softwareOperandIndex >= 1 && softwareOperandIndex <= 3)
      return softwareOperandIndex - 1;
    return std::nullopt;
  }
  llvm_unreachable("unknown resource kind");
}

std::optional<unsigned>
hardwareResultIndexForSoftwareEndpoint(ResourceKind kind,
                                       unsigned softwareResultIndex) {
  switch (kind) {
  case ResourceKind::FabricOp:
    return softwareResultIndex;
  case ResourceKind::MemLoad:
    if (softwareResultIndex <= 1)
      return softwareResultIndex;
    return std::nullopt;
  case ResourceKind::MemStore:
    if (softwareResultIndex == 0)
      return 0;
    return std::nullopt;
  }
  llvm_unreachable("unknown resource kind");
}

HardwareTopology buildHardwareTopology(
    mlir::Operation *hardware, llvm::StringRef hardwareName,
    const llvm::SmallVectorImpl<HardwareResource> &resources) {
  HardwareTopology topology;
  std::map<EndpointKey, std::string> operandEndpoints;
  std::map<EndpointKey, std::string> resultEndpoints;
  llvm::DenseMap<mlir::Operation *, std::string> fabricBoundaryIds;
  for (const HardwareResource &resource : resources) {
    if (resource.kind == ResourceKind::FabricOp && resource.op)
      addGenericEndpointMaps(resource.op, resource.id, operandEndpoints,
                             resultEndpoints);
  }

  llvm::StringMap<unsigned> fabricBoundaryCounts;
  llvm::StringMap<unsigned> routeResourceCounts;
  hardware->walk([&](mlir::Operation *op) {
    llvm::StringRef opName = op->getName().getStringRef();
    if (isFabricBoundaryOp(opName)) {
      unsigned index = fabricBoundaryCounts[opName]++;
      std::string resourceId =
          (hardwareName + "::" + opName + "#" + llvm::Twine(index)).str();
      fabricBoundaryIds.try_emplace(op, resourceId);
      addGenericEndpointMaps(op, resourceId, operandEndpoints, resultEndpoints);
      return;
    }
    if (opName == "fabric.mem") {
      addMemEndpointMaps(op, hardwareName, operandEndpoints, resultEndpoints);
      return;
    }
    if (!isRouteResourceOp(opName))
      return;
    std::string kind = routeResourceKind(opName);
    unsigned index = routeResourceCounts[kind]++;
    addGenericEndpointMaps(
        op, (hardwareName + "::" + kind + "#" + llvm::Twine(index)).str(),
        operandEndpoints, resultEndpoints);
  });

  unsigned ssaEdgeIndex = 0;
  hardware->walk([&](mlir::Operation *op) {
    llvm::StringRef opName = op->getName().getStringRef();
    for (unsigned operandIndex = 0; operandIndex < op->getNumOperands();
         ++operandIndex) {
      mlir::Value operand = op->getOperand(operandIndex);
      std::optional<std::string> sourceEndpoint =
          sourceEndpointForValue(operand, resultEndpoints);
      if (!sourceEndpoint)
        continue;
      auto destIt = operandEndpoints.find(EndpointKey{op, operandIndex});
      if (destIt == operandEndpoints.end())
        continue;
      addTopologySegment(
          topology,
          HardwareRouteSegment{
              "resource_edge", *sourceEndpoint, destIt->second,
              (hardwareName + "::ssa_edge#" + llvm::Twine(ssaEdgeIndex++))
                  .str()});
    }

    addYieldBoundarySegments(op, fabricBoundaryIds, resultEndpoints, topology);
    addFuToPeBoundarySegments(op, fabricBoundaryIds, resultEndpoints, topology);

    if (!isRouteResourceOp(opName))
      return;
    std::optional<std::string> destId;
    auto firstOperand = operandEndpoints.find(EndpointKey{op, 0});
    if (firstOperand != operandEndpoints.end()) {
      llvm::StringRef endpoint = firstOperand->second;
      std::size_t dot = endpoint.rfind(".operand");
      if (dot != std::string::npos)
        destId = endpoint.take_front(dot).str();
    }
    if (!destId)
      return;
    llvm::SmallVector<std::string, 4> switchRows;
    if (opName == "fabric.switch")
      switchRows = switchConnectivityRows(op);
    for (unsigned operandIndex = 0; operandIndex < op->getNumOperands();
         ++operandIndex) {
      for (unsigned resultIndex = 0; resultIndex < op->getNumResults();
           ++resultIndex) {
        if (opName == "fabric.switch" &&
            !switchConnectsInputToOutput(switchRows, operandIndex, resultIndex,
                                         op->getNumOperands()))
          continue;
        addTopologySegment(
            topology,
            HardwareRouteSegment{internalRouteSegmentKind(opName),
                                 endpointFor(*destId, "operand", operandIndex),
                                 endpointFor(*destId, "result", resultIndex),
                                 *destId});
      }
    }
  });
  return topology;
}

llvm::Expected<HardwareModel> collectHardwareModel(mlir::Operation *hardware,
                                                   llvm::StringRef name) {
  HardwareModel model;
  unsigned fabricOpIndex = 0;
  hardware->walk([&](mlir::Operation *op) {
    llvm::StringRef opName = op->getName().getStringRef();
    if (opName == "fabric.op") {
      appendFabricOpResource(op, name, fabricOpIndex++, model.resources);
      return;
    }
    if (opName == "fabric.mem")
      appendMemResources(op, name, model.resources);
  });
  if (model.resources.empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "hardware has no mappable resources");
  model.topology = buildHardwareTopology(hardware, name, model.resources);
  return model;
}

bool configValuesMatch(llvm::StringRef key, llvm::StringRef hardwareValue,
                       llvm::StringRef softwareValue) {
  if (key == "const_hex_value") {
    std::optional<std::string> hardwareHex = canonicalHexValue(hardwareValue);
    std::optional<std::string> softwareHex = canonicalHexValue(softwareValue);
    if (hardwareHex && softwareHex)
      return *hardwareHex == *softwareHex;
  }
  return hardwareValue == softwareValue;
}

std::optional<std::string>
resolvedSoftwareConfigValue(const HardwareResource &resource,
                            llvm::StringRef key, llvm::StringRef value) {
  auto fixed = resource.swConfigs.find(key.str());
  if (fixed != resource.swConfigs.end()) {
    if (configValuesMatch(key, fixed->second, value))
      return fixed->second;
    return std::nullopt;
  }
  auto allowed = resource.hwParamOptions.find(key.str());
  if (allowed == resource.hwParamOptions.end() || allowed->second.empty())
    return value.str();
  for (const std::string &allowedValue : allowed->second) {
    if (configValuesMatch(key, allowedValue, value))
      return allowedValue;
  }
  return std::nullopt;
}

bool resourceSupportsConfig(const HardwareResource &resource,
                            llvm::StringRef key, llvm::StringRef value) {
  return resolvedSoftwareConfigValue(resource, key, value).has_value();
}

bool resourceSupportsSoftwareConfigs(const SoftwareNode &node,
                                     const HardwareResource &resource) {
  for (const auto &[key, value] : softwareConfigsFor(node)) {
    if (!resourceSupportsConfig(resource, key, value))
      return false;
  }
  return true;
}

HardwareResource *
claimResource(SoftwareNode &node,
              llvm::MutableArrayRef<HardwareResource> resources) {
  for (HardwareResource &resource : resources) {
    if (resource.used || resource.kind != node.resourceKind)
      continue;
    if (resource.kind == ResourceKind::FabricOp &&
        !resource.supportedOps.contains(node.operation))
      continue;
    if (!resourceSupportsSoftwareConfigs(node, resource))
      continue;
    resource.used = true;
    return &resource;
  }
  return nullptr;
}

bool resourceIsCompatible(const SoftwareNode &node,
                          const HardwareResource &resource) {
  if (resource.kind != node.resourceKind)
    return false;
  if (resource.kind == ResourceKind::FabricOp &&
      !resource.supportedOps.contains(node.operation))
    return false;
  return resourceSupportsSoftwareConfigs(node, resource);
}

std::optional<std::string> configFor(const HardwareResource &resource,
                                     llvm::StringRef key) {
  auto it = resource.swConfigs.find(key.str());
  if (it == resource.swConfigs.end())
    return std::nullopt;
  return it->second;
}

llvm::DenseMap<mlir::Operation *, std::string>
indexNodeIds(llvm::ArrayRef<SoftwareNode> nodes) {
  llvm::DenseMap<mlir::Operation *, std::string> byOperation;
  for (const SoftwareNode &node : nodes)
    byOperation.try_emplace(node.op, node.id);
  return byOperation;
}

struct RouteBuilder {
  struct ProducerRef {
    std::string softwareId;
    unsigned resultIndex = 0;
  };

  struct EdgeKey {
    std::string producerSoftwareId;
    unsigned producerResultIndex = 0;
    std::string consumerSoftwareId;
    unsigned consumerOperandIndex = 0;

    bool operator<(const EdgeKey &other) const {
      return std::tie(producerSoftwareId, producerResultIndex,
                      consumerSoftwareId, consumerOperandIndex) <
             std::tie(other.producerSoftwareId, other.producerResultIndex,
                      other.consumerSoftwareId, other.consumerOperandIndex);
    }
  };

  llvm::DenseMap<mlir::Value, ProducerRef> producer;
  llvm::DenseMap<mlir::Value, mlir::Value> adapterForward;
  std::map<EdgeKey, std::string> payloadKindByEdge;

  std::optional<ProducerRef> resolve(mlir::Value value) {
    auto direct = producer.find(value);
    if (direct != producer.end())
      return direct->second;
    auto adapter = adapterForward.find(value);
    if (adapter == adapterForward.end())
      return std::nullopt;
    return resolve(adapter->second);
  }
};

std::string payloadKind(mlir::Value value) {
  if (mlir::isa<mlir::NoneType>(value.getType()))
    return "control";
  return "data";
}

void collectValueProducers(
    mlir::Operation *graph,
    const llvm::DenseMap<mlir::Operation *, std::string> &nodeIds,
    RouteBuilder &builder) {
  for (mlir::Operation &op : graph->getRegion(0).front()) {
    auto nodeIt = nodeIds.find(&op);
    if (nodeIt != nodeIds.end()) {
      unsigned resultIndex = 0;
      for (mlir::Value result : op.getResults())
        builder.producer.try_emplace(
            result, RouteBuilder::ProducerRef{nodeIt->second, resultIndex++});
      continue;
    }
    if (isPointerBookkeepingOp(&op))
      continue;
    if (!isAdapterOp(&op) || op.getNumOperands() == 0)
      continue;
    mlir::Value source = op.getOperand(0);
    for (mlir::Value result : op.getResults())
      builder.adapterForward.try_emplace(result, source);
  }
}

bool findRouteDfs(const HardwareTopology &topology,
                  const std::string &currentEndpoint,
                  const std::string &targetEndpoint, llvm::StringSet<> &visited,
                  llvm::SmallVectorImpl<RouteSegment> &path) {
  if (currentEndpoint == targetEndpoint)
    return true;
  auto segmentsIt = topology.segmentsBySource.find(currentEndpoint);
  if (segmentsIt == topology.segmentsBySource.end())
    return false;
  for (const HardwareRouteSegment &candidate : segmentsIt->second) {
    if (!visited.insert(candidate.sinkEndpoint).second)
      continue;
    RouteSegment segment;
    segment.segmentKind = candidate.segmentKind;
    segment.sourceEndpoint = candidate.sourceEndpoint;
    segment.sinkEndpoint = candidate.sinkEndpoint;
    segment.hardwareRef = candidate.hardwareRef;
    path.push_back(std::move(segment));
    if (findRouteDfs(topology, candidate.sinkEndpoint, targetEndpoint, visited,
                     path))
      return true;
    path.pop_back();
  }
  return false;
}

std::optional<llvm::SmallVector<RouteSegment, 2>>
findRoute(const HardwareTopology &topology, const std::string &sourceEndpoint,
          const std::string &sinkEndpoint) {
  llvm::StringSet<> visited;
  visited.insert(sourceEndpoint);
  llvm::SmallVector<RouteSegment, 2> path;
  if (!findRouteDfs(topology, sourceEndpoint, sinkEndpoint, visited, path))
    return std::nullopt;
  for (auto [index, segment] : llvm::enumerate(path))
    segment.segmentId = "seg" + std::to_string(index);
  return path;
}

std::string edgeRefFor(const RouteBuilder::EdgeKey &edge) {
  return edge.producerSoftwareId + ".result" +
         std::to_string(edge.producerResultIndex) + "->" +
         edge.consumerSoftwareId + ".operand" +
         std::to_string(edge.consumerOperandIndex);
}

void addUnroutedEdge(RouteCollection &collection,
                     const RouteBuilder::EdgeKey &edge,
                     llvm::StringRef payloadKind,
                     llvm::StringRef sourceEndpoint,
                     llvm::StringRef sinkEndpoint,
                     llvm::StringRef diagnostic) {
  UnroutedEdgeRecord record;
  record.edgeRef = edgeRefFor(edge);
  record.producerBinding = "placement:" + edge.producerSoftwareId;
  record.consumerBinding = "placement:" + edge.consumerSoftwareId;
  record.payloadKind = payloadKind.str();
  record.fromSoftwareId = edge.producerSoftwareId;
  record.toSoftwareId = edge.consumerSoftwareId;
  record.sourceEndpoint = sourceEndpoint.str();
  record.sinkEndpoint = sinkEndpoint.str();
  record.diagnostic = diagnostic.str();
  collection.unroutedEdgeDetails.push_back(std::move(record));
  ++collection.unroutedEdges;
}

RouteCollection collectRoutes(llvm::ArrayRef<SoftwareNode> nodes,
                              mlir::Operation *graph,
                              llvm::ArrayRef<PlacementRecord> placements,
                              const HardwareTopology &topology) {
  RouteBuilder builder;
  llvm::DenseMap<mlir::Operation *, std::string> nodeIds = indexNodeIds(nodes);
  collectValueProducers(graph, nodeIds, builder);

  llvm::StringMap<std::string> hardwareBySoftware;
  for (const PlacementRecord &placement : placements)
    hardwareBySoftware.try_emplace(placement.softwareId, placement.hardwareId);
  std::map<std::string, ResourceKind> kindBySoftware;
  for (const SoftwareNode &node : nodes)
    kindBySoftware.try_emplace(node.id, node.resourceKind);

  for (const SoftwareNode &node : nodes) {
    unsigned operandIndex = 0;
    for (mlir::Value operand : node.op->getOperands()) {
      if (!hardwareOperandIndexForSoftwareEndpoint(node.resourceKind,
                                                   operandIndex)) {
        ++operandIndex;
        continue;
      }
      std::optional<RouteBuilder::ProducerRef> source =
          builder.resolve(operand);
      if (!source || source->softwareId == node.id) {
        ++operandIndex;
        continue;
      }
      RouteBuilder::EdgeKey key{source->softwareId, source->resultIndex,
                                node.id, operandIndex++};
      builder.payloadKindByEdge.try_emplace(std::move(key),
                                            payloadKind(operand));
    }
  }

  RouteCollection collection;
  std::size_t index = 0;
  for (const auto &[edge, kind] : builder.payloadKindByEdge) {
    const std::string &from = edge.producerSoftwareId;
    const std::string &to = edge.consumerSoftwareId;
    auto fromHw = hardwareBySoftware.find(from);
    auto toHw = hardwareBySoftware.find(to);
    if (fromHw == hardwareBySoftware.end() || toHw == hardwareBySoftware.end())
      continue;
    auto fromKind = kindBySoftware.find(from);
    auto toKind = kindBySoftware.find(to);
    if (fromKind == kindBySoftware.end() || toKind == kindBySoftware.end())
      continue;
    std::optional<unsigned> producerResultIndex =
        hardwareResultIndexForSoftwareEndpoint(fromKind->second,
                                               edge.producerResultIndex);
    std::optional<unsigned> consumerOperandIndex =
        hardwareOperandIndexForSoftwareEndpoint(toKind->second,
                                                edge.consumerOperandIndex);
    if (!producerResultIndex || !consumerOperandIndex) {
      addUnroutedEdge(collection, edge, kind, "", "",
                      "software endpoint has no hardware endpoint");
      continue;
    }
    std::string sourceEndpoint =
        endpointFor(fromHw->second, "result", *producerResultIndex);
    std::string sinkEndpoint =
        endpointFor(toHw->second, "operand", *consumerOperandIndex);
    std::optional<llvm::SmallVector<RouteSegment, 2>> path =
        findRoute(topology, sourceEndpoint, sinkEndpoint);
    if (!path) {
      addUnroutedEdge(collection, edge, kind, sourceEndpoint, sinkEndpoint,
                      "no Fabric ADG route connects source to sink");
      continue;
    }
    std::string recordId = "route#" + std::to_string(index++);
    std::string edgeRef = edgeRefFor(edge);
    RouteRecord route;
    route.recordId = recordId;
    route.edgeRef = edgeRef;
    route.producerBinding = "placement:" + from;
    route.consumerBinding = "placement:" + to;
    route.payloadKind = kind;
    route.fromSoftwareId = from;
    route.toSoftwareId = to;
    route.segments = std::move(*path);
    collection.routes.push_back(std::move(route));
  }
  return collection;
}

bool partialPlacementRoutes(llvm::ArrayRef<SoftwareNode> nodes,
                            mlir::Operation *graph,
                            llvm::ArrayRef<PlacementRecord> placements,
                            const HardwareTopology &topology) {
  RouteCollection partial = collectRoutes(nodes, graph, placements, topology);
  return partial.unroutedEdges == 0;
}

bool chooseRouteFeasiblePlacements(
    llvm::MutableArrayRef<SoftwareNode> nodes,
    llvm::MutableArrayRef<HardwareResource> resources, mlir::Operation *graph,
    const HardwareTopology &topology,
    llvm::SmallVectorImpl<PlacementRecord> &placements, unsigned nodeIndex) {
  if (nodeIndex == nodes.size())
    return true;

  SoftwareNode &node = nodes[nodeIndex];
  for (HardwareResource &resource : resources) {
    if (resource.used || !resourceIsCompatible(node, resource))
      continue;
    resource.used = true;
    placements.push_back(PlacementRecord{
        node.id, node.operation, resourceKindName(node.resourceKind).str(),
        resource.id, resource.schedule});

    if (partialPlacementRoutes(nodes, graph, placements, topology) &&
        chooseRouteFeasiblePlacements(nodes, resources, graph, topology,
                                      placements, nodeIndex + 1))
      return true;

    placements.pop_back();
    resource.used = false;
  }
  return false;
}

bool placeRouteFeasible(llvm::MutableArrayRef<SoftwareNode> nodes,
                        llvm::MutableArrayRef<HardwareResource> resources,
                        mlir::Operation *graph,
                        const HardwareTopology &topology,
                        llvm::SmallVectorImpl<PlacementRecord> &placements) {
  for (HardwareResource &resource : resources)
    resource.used = false;
  placements.clear();
  if (chooseRouteFeasiblePlacements(nodes, resources, graph, topology,
                                    placements, 0))
    return true;
  for (HardwareResource &resource : resources)
    resource.used = false;
  placements.clear();
  return false;
}

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

std::string routeTarget(const MappingSummary &summary,
                        llvm::StringRef recordId) {
  return summary.mappingId + "::" + recordId.str();
}

std::string routeSource(const RouteRecord &route) {
  return "route:" + route.recordId;
}

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

llvm::json::Object configJson(const ConfigEntry &entry) {
  return llvm::json::Object{
      {"target", entry.target},
      {"register", entry.registerName},
      {"value", entry.value},
      {"source", entry.source},
  };
}

} // namespace

llvm::Expected<MappingSummary>
loom::pnr::createMapping(const MappingOptions &options) {
  mlir::DialectRegistry registry = makeRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> dfg =
      parseModule(context, options.dfgMlirPath);
  if (!dfg)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "could not parse DFG MLIR");
  mlir::OwningOpRef<mlir::ModuleOp> hardware =
      parseModule(context, options.hardwareMlirPath);
  if (!hardware)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "could not parse hardware MLIR");

  mlir::Operation *graph =
      findSymbolOp(*dfg, "dataflow.graph.func", options.graphName);
  if (!graph)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "could not find dataflow graph %s",
                                   options.graphName.c_str());
  mlir::Operation *hardwareOp =
      findSymbolOp(*hardware, "fabric.module", options.hardwareName);
  if (!hardwareOp)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "could not find fabric hardware %s",
                                   options.hardwareName.c_str());

  auto nodesOrErr = collectSoftwareNodes(graph);
  if (!nodesOrErr)
    return nodesOrErr.takeError();
  auto hardwareModelOrErr =
      collectHardwareModel(hardwareOp, options.hardwareName);
  if (!hardwareModelOrErr)
    return hardwareModelOrErr.takeError();

  MappingSummary summary;
  summary.workload =
      options.workload.empty() ? options.graphName : options.workload;
  summary.hardware = options.hardwareName;
  summary.graph = options.graphName;
  summary.mappingId =
      mappingId(summary.workload, summary.graph, summary.hardware);
  loom::ResolvedConfig resolvedConfig = loom::defaultResolvedConfig();
  summary.configId = resolvedConfig.configId;
  summary.configFingerprint = loom::resolvedConfigFingerprint(resolvedConfig);
  summary.componentConfigView = "pnr.mapping.v1";
  summary.componentConfigFingerprint = loom::componentConfigFingerprint(
      resolvedConfig, summary.componentConfigView);
  summary.status = "pass";

  if (!placeRouteFeasible(*nodesOrErr, hardwareModelOrErr->resources, graph,
                          hardwareModelOrErr->topology, summary.placements)) {
    for (HardwareResource &resource : hardwareModelOrErr->resources)
      resource.used = false;
    for (SoftwareNode &node : *nodesOrErr) {
      HardwareResource *resource =
          claimResource(node, hardwareModelOrErr->resources);
      if (!resource) {
        summary.status = "fail";
        summary.diagnostic =
            "missing hardware resource for software op " + node.operation;
        ++summary.unplacedRecords;
        continue;
      }
      summary.placements.push_back(PlacementRecord{
          node.id, node.operation, resourceKindName(node.resourceKind).str(),
          resource->id, resource->schedule});
    }
  }

  llvm::StringMap<const SoftwareNode *> nodeById;
  for (const SoftwareNode &node : *nodesOrErr)
    nodeById.try_emplace(node.id, &node);
  llvm::StringMap<const HardwareResource *> resourceById;
  for (const HardwareResource &resource : hardwareModelOrErr->resources)
    resourceById.try_emplace(resource.id, &resource);
  for (const PlacementRecord &placement : summary.placements) {
    auto nodeIt = nodeById.find(placement.softwareId);
    auto resourceIt = resourceById.find(placement.hardwareId);
    if (nodeIt == nodeById.end() || resourceIt == resourceById.end())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "mapping placement references unknown software or hardware id");
    if (llvm::Error err =
            appendPlacementConfig(summary, *nodeIt->second, *resourceIt->second))
      return std::move(err);
  }

  RouteCollection routeCollection = collectRoutes(
      *nodesOrErr, graph, summary.placements, hardwareModelOrErr->topology);
  summary.routes = std::move(routeCollection.routes);
  summary.unroutedEdgeDetails =
      std::move(routeCollection.unroutedEdgeDetails);
  summary.unroutedEdges = routeCollection.unroutedEdges;
  if (summary.status == "pass" && summary.unroutedEdges != 0) {
    summary.status = "fail";
    summary.diagnostic = "unrouted software edges lack Fabric ADG connectivity";
  }
  if (summary.status == "pass") {
    appendRouteConfig(summary);
    if (llvm::Error err = validateConfigBitstream(summary))
      return std::move(err);
    summary.diagnostic = "mapped software graph to fabric resources";
  } else {
    summary.configEntries.clear();
  }
  return summary;
}

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

  llvm::json::Object root{
      {"schema_version", 1},
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
