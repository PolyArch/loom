#include "MemoryAccess.h"
#include "Support.h"

#include "Fabric/IR/MemoryServiceContract.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <utility>

namespace loom::hardware::rtl::hierarchy {
namespace {

/// The finite address-width inventory of one access class. The support
/// ceiling of a lane width is not decided here: the portable address
/// arithmetic derived from the complete layout rejects a module whose widest
/// lane exceeds the byte-address domain before any access case is built.
llvm::Expected<std::vector<std::uint32_t>>
finiteWidths(const ::fabric::MemoryAccessClass &access) {
  std::vector<std::uint32_t> result;
  if (const auto *widths = access.rootRelativeIndexWidths()) {
    for (const ::fabric::UnsignedInterval interval : widths->intervals()) {
      if (interval.upper - interval.lower > 8)
        return unsupported("portable memory address-width domain is too wide");
      for (std::uint64_t width = interval.lower; width <= interval.upper;
           ++width)
        result.push_back(static_cast<std::uint32_t>(width));
    }
  } else {
    const auto *formats = access.addressPointerFormats();
    if (!formats)
      return invalid("pointer-addressed memory access has no format domain");
    for (const ::fabric::PointerFormat &format : formats->formats()) {
      if (format.representationBits == 0)
        return invalid("portable memory pointer representation is empty");
      result.push_back(format.representationBits);
    }
  }
  llvm::sort(result);
  result.erase(std::unique(result.begin(), result.end()), result.end());
  if (result.empty())
    return invalid("memory access has an empty address-width domain");
  return result;
}

} // namespace

llvm::Expected<std::vector<MemoryAccessCase>>
deriveMemoryAccessCases(const fabric::FabricArtifactView &fabric,
                  fabric::FabricMemoryOccurrenceRef memory) {
  std::vector<MemoryAccessCase> result;
  for (auto [portOrdinal, portRef] :
       llvm::enumerate(fabric.memoryOperationPorts(memory))) {
    const auto *port = fabric.memoryOperationPort(portRef);
    if (!port)
      return invalid("memory operation port does not resolve");
    for (auto [capabilityOrdinal, capability] :
         llvm::enumerate(port->capabilityAlternatives())) {
      const auto schema = capability.actorContractDomain.actorSchema();
      const bool read = schema == ::dataflow::OperationSchemaId::DataflowLoad;
      if (!read && schema != ::dataflow::OperationSchemaId::DataflowStore)
        return unsupported(
            "portable memory profile admits only plain load and store");
      if (!capability.accessDomain)
        return unsupported(
            "portable load/store capability has no access domain");
      for (auto [classOrdinal, access] :
           llvm::enumerate(capability.accessDomain->accessClasses())) {
        auto addressWidths = finiteWidths(access);
        if (!addressWidths)
          return addressWidths.takeError();
        for (const ::fabric::UnsignedInterval interval :
             access.elementWidths().intervals()) {
          if (interval.lower == 0 || interval.upper > 4096 ||
              interval.upper - interval.lower > 64)
            return unsupported(
                "portable memory element-width domain is too wide");
          for (std::uint64_t elementWidth = interval.lower;
               elementWidth <= interval.upper; ++elementWidth) {
            if (elementWidth % 8 != 0)
              continue;
            MemoryAccessCase selected;
            selected.physicalPort = static_cast<std::uint32_t>(portOrdinal);
            selected.capability = static_cast<std::uint32_t>(capabilityOrdinal);
            selected.accessClass = static_cast<std::uint32_t>(classOrdinal);
            selected.read = read;
            selected.accessForm = access.accessForm();
            selected.addressForm = access.addressForm();
            selected.elementWidthBits = elementWidth;
            selected.addressWidths = *addressWidths;
            selected.laneCounts.assign(
                access.flattenedLaneCounts().intervals().begin(),
                access.flattenedLaneCounts().intervals().end());
            for (const ::fabric::MaskInactivePair pair :
                 access.maskInactivePairs())
              selected.dynamicMasks.push_back(
                  pair.mask == ::dataflow::semantics::MemoryMaskForm::Dynamic);
            result.push_back(std::move(selected));
          }
        }
      }
    }
  }
  return result;
}

llvm::Expected<std::vector<MemoryAccessCase>>
deriveLocalMemoryServiceAccessCases(const fabric::FabricArtifactView &fabric,
                              fabric::FabricMemoryOccurrenceRef memory) {
  std::vector<MemoryAccessCase> result;
  const auto *service = fabric.localMemoryService(memory);
  if (!service)
    return result;
  for (auto [capabilityOrdinal, capability] :
       llvm::enumerate(service->capabilities())) {
    const auto schema = capability.actorContractDomain.actorSchema();
    const bool read = schema == ::dataflow::OperationSchemaId::DataflowLoad;
    if (!read && schema != ::dataflow::OperationSchemaId::DataflowStore)
      return unsupported(
          "portable local service admits only plain load and store");
    if (!capability.accessDomain)
      return unsupported("portable local load/store has no access domain");
    for (auto [classOrdinal, access] :
         llvm::enumerate(capability.accessDomain->accessClasses())) {
      auto addressWidths = finiteWidths(access);
      if (!addressWidths)
        return addressWidths.takeError();
      for (const ::fabric::UnsignedInterval interval :
           access.elementWidths().intervals()) {
        if (interval.lower == 0 || interval.upper > 4096 ||
            interval.upper - interval.lower > 64)
          return unsupported(
              "portable local-service element-width domain is too wide");
        for (std::uint64_t elementWidth = interval.lower;
             elementWidth <= interval.upper; ++elementWidth) {
          if (elementWidth % 8 != 0)
            continue;
          MemoryAccessCase selected;
          selected.capability = static_cast<std::uint32_t>(capabilityOrdinal);
          selected.accessClass = static_cast<std::uint32_t>(classOrdinal);
          selected.read = read;
          selected.accessForm = access.accessForm();
          selected.addressForm = access.addressForm();
          selected.elementWidthBits = elementWidth;
          selected.addressWidths = *addressWidths;
          selected.laneCounts.assign(
              access.flattenedLaneCounts().intervals().begin(),
              access.flattenedLaneCounts().intervals().end());
          for (const ::fabric::MaskInactivePair pair :
               access.maskInactivePairs())
            selected.dynamicMasks.push_back(
                pair.mask == ::dataflow::semantics::MemoryMaskForm::Dynamic);
          for (std::uint64_t regionOrdinal : capability.serviceRegionOrdinals) {
            const auto &region = service->regions()[regionOrdinal];
            if (region.behavior !=
                ::fabric::MemoryServiceRegionBehavior::Storage)
              return unsupported(
                  "portable local-memory profile does not implement MMIO "
                  "regions");
            selected.storageRegions.push_back(
                {region.addressBaseBytes,
                 region.addressBaseBytes + region.sizeBytes - 1});
          }
          result.push_back(std::move(selected));
        }
      }
    }
  }
  return result;
}

std::uint64_t
accessFormCode(::dataflow::semantics::MemoryAccessForm accessForm) {
  switch (accessForm) {
  case ::dataflow::semantics::MemoryAccessForm::Element:
    return 0;
  case ::dataflow::semantics::MemoryAccessForm::Contiguous:
    return 1;
  case ::dataflow::semantics::MemoryAccessForm::Indexed:
    return 2;
  }
  llvm_unreachable("unknown memory access form");
}

std::uint64_t
addressFormCode(::dataflow::semantics::MemoryAddressForm addressForm) {
  switch (addressForm) {
  case ::dataflow::semantics::MemoryAddressForm::RootRelative:
    return 0;
  case ::dataflow::semantics::MemoryAddressForm::PointerAddressed:
    return 1;
  }
  llvm_unreachable("unknown memory address form");
}

} // namespace loom::hardware::rtl::hierarchy
