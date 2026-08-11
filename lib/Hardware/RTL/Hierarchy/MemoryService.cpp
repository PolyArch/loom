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

using Role = ::dataflow::semantics::ServiceValueRole;

std::optional<std::uint64_t> maximum(const ::fabric::UnsignedDomain &domain) {
  if (domain.intervals().empty())
    return std::nullopt;
  return domain.intervals().back().upper;
}

llvm::Error updateMaximum(std::uint32_t &destination, std::uint64_t value,
                          llvm::StringRef description) {
  if (value == 0)
    return invalid(description + " has zero width");
  if (value > mlir::IntegerType::kMaxWidth ||
      value > std::numeric_limits<std::uint32_t>::max())
    return unsupported(description + " exceeds CIRCT integer capacity");
  destination = std::max(destination, static_cast<std::uint32_t>(value));
  return llvm::Error::success();
}

llvm::Error updateFromAccess(PortableMemoryServiceLayout &layout,
                             const ::fabric::MemoryAccessClass &access) {
  const auto element = maximum(access.elementWidths());
  const auto lanes = maximum(access.flattenedLaneCounts());
  if (!element || !lanes)
    return invalid("portable memory access has an empty geometry domain");
  if (*element > std::numeric_limits<std::uint64_t>::max() / *lanes)
    return unsupported("portable memory data carrier width overflows u64");
  if (llvm::Error error = updateMaximum(layout.dataWidthBits, *element * *lanes,
                                        "portable memory data carrier"))
    return error;

  const bool hasDynamicMask = llvm::any_of(
      access.maskInactivePairs(), [](const ::fabric::MaskInactivePair &pair) {
        return pair.mask == ::dataflow::semantics::MemoryMaskForm::Dynamic;
      });
  if (hasDynamicMask)
    if (llvm::Error error = updateMaximum(layout.maskWidthBits, *lanes,
                                          "portable memory mask carrier"))
      return error;

  std::uint64_t laneWidth = 0;
  if (const auto *widths = access.rootRelativeIndexWidths()) {
    const auto width = maximum(*widths);
    if (!width)
      return invalid("portable memory address domain is empty");
    laneWidth = *width;
  } else if (const auto *formats = access.addressPointerFormats()) {
    for (const ::fabric::PointerFormat &format : formats->formats())
      laneWidth = std::max<std::uint64_t>(laneWidth, format.representationBits);
  }
  if (laneWidth == 0)
    return invalid("portable memory access has no address representation");
  const std::uint64_t addressLanes =
      access.accessForm() == ::dataflow::semantics::MemoryAccessForm::Indexed
          ? *lanes
          : 1;
  if (laneWidth > std::numeric_limits<std::uint64_t>::max() / addressLanes)
    return unsupported("portable memory address carrier width overflows u64");
  return updateMaximum(layout.addressWidthBits, laneWidth * addressLanes,
                       "portable memory address carrier");
}

llvm::Error updateFromAccessDomain(
    PortableMemoryServiceLayout &layout,
    const ::fabric::ParameterizedMemoryAccessDomain &domain) {
  for (const ::fabric::MemoryAccessClass &access : domain.accessClasses())
    if (llvm::Error error = updateFromAccess(layout, access))
      return error;
  return llvm::Error::success();
}

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

llvm::Expected<PortableMemoryServiceLayout>
derivePortableMemoryServiceLayout(const fabric::FabricArtifactView &fabric) {
  PortableMemoryServiceLayout layout;
  for (fabric::FabricMemoryOccurrenceRef memory : fabric.memoryOccurrences()) {
    for (fabric::FabricMemoryOperationPortRef portRef :
         fabric.memoryOperationPorts(memory)) {
      const auto *port = fabric.memoryOperationPort(portRef);
      if (!port)
        return invalid("portable memory operation port does not resolve");
      for (const ::fabric::MemoryCapabilityAlternativeRecord &capability :
           port->capabilityAlternatives()) {
        for (const ::fabric::MemoryRoleEndpointBindingRecord &binding :
             capability.roleToEndpoint) {
          const auto dataPath = fabric.transportEndpointDataPath(
              fabric::FabricTransportEndpointRef{
                  fabric::FabricTransportEndpointOwnerRef::of(memory),
                  binding.endpointOrdinal});
          if (!dataPath)
            return invalid("portable memory role endpoint does not resolve");
          switch (binding.role) {
          case Role::Address:
            if (llvm::Error error = updateMaximum(
                    layout.addressWidthBits, dataPath->payloadWidthBits,
                    "portable memory address endpoint"))
              return std::move(error);
            break;
          case Role::Data:
          case Role::Update:
          case Role::Expected:
          case Role::Desired:
          case Role::Old:
            if (llvm::Error error = updateMaximum(
                    layout.dataWidthBits, dataPath->payloadWidthBits,
                    "portable memory data endpoint"))
              return std::move(error);
            break;
          case Role::Mask:
            if (llvm::Error error = updateMaximum(
                    layout.maskWidthBits, dataPath->payloadWidthBits,
                    "portable memory mask endpoint"))
              return std::move(error);
            break;
          default:
            break;
          }
        }
        if (capability.accessDomain)
          if (llvm::Error error =
                  updateFromAccessDomain(layout, *capability.accessDomain))
            return std::move(error);
      }
    }
    if (const auto *service = fabric.localMemoryService(memory))
      for (const ::fabric::MemoryServiceCapabilityDeclaration &capability :
           service->capabilities())
        if (capability.accessDomain)
          if (llvm::Error error =
                  updateFromAccessDomain(layout, *capability.accessDomain))
            return std::move(error);
  }
  if (layout.addressWidthBits == 0)
    layout.addressWidthBits = 1;
  if (layout.dataWidthBits == 0)
    layout.dataWidthBits = 1;
  if (layout.maskWidthBits == 0)
    layout.maskWidthBits = 1;
  return layout;
}

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
