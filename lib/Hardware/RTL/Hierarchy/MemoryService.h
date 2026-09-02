#ifndef LOOM_LIB_HARDWARE_RTL_HIERARCHY_MEMORYSERVICE_H
#define LOOM_LIB_HARDWARE_RTL_HIERARCHY_MEMORYSERVICE_H

#include "Fabric/Identity/FabricRefImport.h"
#include "Hardware/RTL/MemoryServiceTransport.h"

#include "circt/Dialect/HW/HWOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom::hardware::rtl::hierarchy {

struct MemoryServicePortPlan final {
  std::string name;
  fabric::FabricMemoryEndpointRole role =
      fabric::FabricMemoryEndpointRole::Manager;
  circt::hw::PortInfo requestKind;
  circt::hw::PortInfo requestAddress;
  circt::hw::PortInfo requestData;
  circt::hw::PortInfo requestMask;
  circt::hw::PortInfo requestActiveLanesKind;
  circt::hw::PortInfo requestAccessForm;
  circt::hw::PortInfo requestAddressForm;
  circt::hw::PortInfo requestElementWidth;
  circt::hw::PortInfo requestLaneCount;
  circt::hw::PortInfo requestAddressLaneWidth;
  circt::hw::PortInfo requestBaseAddress;
  circt::hw::PortInfo requestContext;
  circt::hw::PortInfo requestValid;
  circt::hw::PortInfo requestReady;
  circt::hw::PortInfo responseData;
  circt::hw::PortInfo responseValid;
  circt::hw::PortInfo responseReady;
};

struct MemoryEndpointPortPlan final {
  fabric::FabricMemoryEndpointRef endpoint;
  MemoryServicePortPlan ports;
};

struct ModuleBoundaryMemoryPortProjection final {
  fabric::FabricModuleBoundaryEndpointRef boundary;
  MemoryServicePortPlan ports;
};

llvm::Expected<std::vector<MemoryEndpointPortPlan>>
deriveMemoryEndpointPortPlans(mlir::OpBuilder &builder,
                              const fabric::FabricArtifactView &fabric,
                              fabric::FabricMemoryOccurrenceRef memory,
                              const PortableMemoryServiceLayout &layout);

llvm::Expected<std::vector<ModuleBoundaryMemoryPortProjection>>
deriveModuleBoundaryMemoryPorts(mlir::OpBuilder &builder,
                                const fabric::FabricArtifactView &fabric,
                                const PortableMemoryServiceLayout &layout);

void appendMemoryServicePorts(
    llvm::SmallVectorImpl<circt::hw::PortInfo> &inputs,
    llvm::SmallVectorImpl<circt::hw::PortInfo> &outputs,
    const MemoryServicePortPlan &ports);

} // namespace loom::hardware::rtl::hierarchy

#endif // LOOM_LIB_HARDWARE_RTL_HIERARCHY_MEMORYSERVICE_H
