#ifndef LOOM_LIB_HARDWARE_RTL_HIERARCHY_MEMORYACCESS_H
#define LOOM_LIB_HARDWARE_RTL_HIERARCHY_MEMORYACCESS_H

#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/IR/MemoryCapabilityDomains.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::hardware::rtl::hierarchy {

struct MemoryAccessCase final {
  std::uint32_t physicalPort = 0;
  std::uint32_t capability = 0;
  std::uint32_t accessClass = 0;
  bool read = false;
  ::dataflow::semantics::MemoryAccessForm accessForm =
      ::dataflow::semantics::MemoryAccessForm::Element;
  ::dataflow::semantics::MemoryAddressForm addressForm =
      ::dataflow::semantics::MemoryAddressForm::RootRelative;
  std::uint64_t elementWidthBits = 0;
  std::vector<std::uint32_t> addressWidths;
  std::vector<::fabric::UnsignedInterval> laneCounts;
  std::vector<bool> dynamicMasks;
  std::vector<::fabric::UnsignedInterval> storageRegions;
};

llvm::Expected<std::vector<MemoryAccessCase>>
deriveMemoryAccessCases(const fabric::FabricArtifactView &fabric,
                        fabric::FabricMemoryOccurrenceRef memory);
llvm::Expected<std::vector<MemoryAccessCase>>
deriveLocalMemoryServiceAccessCases(const fabric::FabricArtifactView &fabric,
                                    fabric::FabricMemoryOccurrenceRef memory);
std::uint64_t accessFormCode(::dataflow::semantics::MemoryAccessForm accessForm);
std::uint64_t addressFormCode(::dataflow::semantics::MemoryAddressForm addressForm);

} // namespace loom::hardware::rtl::hierarchy

#endif
