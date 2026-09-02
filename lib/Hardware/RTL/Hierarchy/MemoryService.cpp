#include "MemoryService.h"

#include "Support.h"

#include "Hardware/RTL/MemoryServiceTransport.h"

#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/IR/MemoryCapabilityDomains.h"
#include "Fabric/IR/MemoryServiceContract.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace loom::hardware::rtl::hierarchy {
namespace {

circt::hw::PortInfo makePort(mlir::OpBuilder &builder, llvm::StringRef name,
                             mlir::Type type,
                             circt::hw::ModulePort::Direction direction) {
  return circt::hw::PortInfo{{builder.getStringAttr(name), type, direction}};
}

MemoryServicePortPlan makePorts(mlir::OpBuilder &builder, std::string name,
                                fabric::FabricMemoryEndpointRole role,
                                const PortableMemoryServiceLayout &layout) {
  const auto requestDirection =
      role == fabric::FabricMemoryEndpointRole::Manager
          ? circt::hw::ModulePort::Direction::Output
          : circt::hw::ModulePort::Direction::Input;
  const auto responseDirection =
      role == fabric::FabricMemoryEndpointRole::Manager
          ? circt::hw::ModulePort::Direction::Input
          : circt::hw::ModulePort::Direction::Output;
  const auto opposite = [](circt::hw::ModulePort::Direction direction) {
    return direction == circt::hw::ModulePort::Direction::Input
               ? circt::hw::ModulePort::Direction::Output
               : circt::hw::ModulePort::Direction::Input;
  };
  const auto port = [&](llvm::StringRef suffix, mlir::Type type,
                        circt::hw::ModulePort::Direction direction) {
    return makePort(builder, name + suffix.str(), type, direction);
  };
  const mlir::Type bit = builder.getI1Type();
  return MemoryServicePortPlan{
      name,
      role,
      port("_request_kind", bit, requestDirection),
      port("_request_address", builder.getIntegerType(layout.addressWidthBits),
           requestDirection),
      port("_request_data", builder.getIntegerType(layout.dataWidthBits),
           requestDirection),
      port("_request_mask", builder.getIntegerType(layout.maskWidthBits),
           requestDirection),
      port("_request_active_lanes_kind", bit, requestDirection),
      port("_request_access_form",
           builder.getIntegerType(portableMemoryAccessFormWidth),
           requestDirection),
      port("_request_address_form", bit, requestDirection),
      port("_request_element_width",
           builder.getIntegerType(portableMemoryElementWidthFieldWidth),
           requestDirection),
      port("_request_lane_count",
           builder.getIntegerType(portableMemoryLaneCountFieldWidth),
           requestDirection),
      port("_request_address_lane_width",
           builder.getIntegerType(portableMemoryAddressLaneWidthFieldWidth),
           requestDirection),
      port("_request_base_address",
           builder.getIntegerType(portableMemoryBaseAddressFieldWidth),
           requestDirection),
      port("_request_context", builder.getIntegerType(portableMemoryContextFieldWidth),
           requestDirection),
      port("_request_valid", bit, requestDirection),
      port("_request_ready", bit, opposite(requestDirection)),
      port("_response_data", builder.getIntegerType(layout.dataWidthBits),
           responseDirection),
      port("_response_valid", bit, responseDirection),
      port("_response_ready", bit, opposite(responseDirection))};
}

} // namespace

llvm::Expected<std::vector<MemoryEndpointPortPlan>>
deriveMemoryEndpointPortPlans(mlir::OpBuilder &builder,
                              const fabric::FabricArtifactView &fabric,
                              fabric::FabricMemoryOccurrenceRef memory,
                              const PortableMemoryServiceLayout &layout) {
  std::vector<MemoryEndpointPortPlan> result;
  const fabric::FabricMemoryEndpointOwnerRef owner =
      fabric::FabricMemoryEndpointOwnerRef::of(memory);
  result.reserve(fabric.memoryEndpointCount(owner));
  for (fabric::FabricOrdinal ordinal = 0;
       ordinal != fabric.memoryEndpointCount(owner); ++ordinal) {
    const fabric::FabricMemoryEndpointRef endpoint{owner, ordinal};
    const auto role = fabric.memoryEndpointRole(endpoint);
    if (!role)
      return invalid("portable memory endpoint has no role");
    result.push_back(
        {endpoint, makePorts(builder, "service_" + std::to_string(ordinal),
                             *role, layout)});
  }
  return result;
}

llvm::Expected<std::vector<ModuleBoundaryMemoryPortProjection>>
deriveModuleBoundaryMemoryPorts(mlir::OpBuilder &builder,
                                const fabric::FabricArtifactView &fabric,
                                const PortableMemoryServiceLayout &layout) {
  const auto module = fabric.moduleRootTemplate();
  if (!module)
    return invalid("memory boundary projection requires one Module root");
  std::vector<ModuleBoundaryMemoryPortProjection> result;
  for (fabric::FabricPortDirection direction :
       {fabric::FabricPortDirection::Input,
        fabric::FabricPortDirection::Output}) {
    const std::uint64_t count =
        fabric.moduleBoundaryEndpointCount(*module, direction);
    for (fabric::FabricOrdinal ordinal = 0; ordinal != count; ++ordinal) {
      const fabric::FabricModuleBoundaryEndpointRef boundary{*module, direction,
                                                             ordinal};
      const auto plane = fabric.moduleBoundaryEndpointPlane(boundary);
      if (!plane)
        return invalid("memory boundary endpoint does not resolve");
      if (*plane != fabric::FabricSpatialAttachmentEndpointRef::Plane::Memory)
        continue;
      const auto role = direction == fabric::FabricPortDirection::Input
                            ? fabric::FabricMemoryEndpointRole::Manager
                            : fabric::FabricMemoryEndpointRole::Subordinate;
      const std::string prefix =
          std::string("memory_") +
          (direction == fabric::FabricPortDirection::Input ? "input_"
                                                           : "output_") +
          std::to_string(ordinal);
      result.push_back({boundary, makePorts(builder, prefix, role, layout)});
    }
  }
  return result;
}

void appendMemoryServicePorts(
    llvm::SmallVectorImpl<circt::hw::PortInfo> &inputs,
    llvm::SmallVectorImpl<circt::hw::PortInfo> &outputs,
    const MemoryServicePortPlan &ports) {
  const auto append = [&](const circt::hw::PortInfo &port) {
    (port.isOutput() ? outputs : inputs).push_back(port);
  };
  append(ports.requestKind);
  append(ports.requestAddress);
  append(ports.requestData);
  append(ports.requestMask);
  append(ports.requestActiveLanesKind);
  append(ports.requestAccessForm);
  append(ports.requestAddressForm);
  append(ports.requestElementWidth);
  append(ports.requestLaneCount);
  append(ports.requestAddressLaneWidth);
  append(ports.requestBaseAddress);
  append(ports.requestContext);
  append(ports.requestValid);
  append(ports.requestReady);
  append(ports.responseData);
  append(ports.responseValid);
  append(ports.responseReady);
}

} // namespace loom::hardware::rtl::hierarchy
