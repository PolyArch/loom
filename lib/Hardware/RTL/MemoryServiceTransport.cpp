#include "Hardware/RTL/MemoryServiceTransport.h"

#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/IR/MemoryCapabilityDomains.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/Identity/FabricRefImport.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <system_error>

namespace loom::hardware::rtl {
namespace {

llvm::Error invalid(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "portable_memory_request_context_invalid: " + detail);
}

llvm::Error unsupported(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::not_supported),
      "portable_memory_request_context_unsupported: " + detail);
}

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
  if (llvm::Error error = updateMaximum(
          layout.maximumAddressLaneWidthBits, laneWidth,
          "portable memory address lane"))
    return error;
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

} // namespace

llvm::Expected<PortableMemoryServiceLayout>
derivePortableMemoryServiceLayout(const fabric::FabricArtifactView &fabric) {
  using Role = ::dataflow::semantics::ServiceValueRole;
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
  if (layout.maximumAddressLaneWidthBits == 0)
    layout.maximumAddressLaneWidthBits = 1;
  return layout;
}

std::optional<PortableMemoryAddressArithmetic>
derivePortableMemoryAddressArithmetic(
    const PortableMemoryServiceLayout &layout) {
  PortableMemoryAddressArithmetic arithmetic;
  if (layout.maximumAddressLaneWidthBits == 0 ||
      layout.maximumAddressLaneWidthBits > arithmetic.byteAddressWidthBits)
    return std::nullopt;
  arithmetic.laneWidthBits = layout.maximumAddressLaneWidthBits;
  return arithmetic;
}

llvm::Expected<PortableMemoryRequestContextIndex>
PortableMemoryRequestContextIndex::get(
    const fabric::FabricArtifactView &fabric) {
  std::uint64_t firstContext = 0;
  std::map<std::uint64_t, Range> ranges;
  for (fabric::FabricMemoryOccurrenceRef candidate :
       fabric.memoryOccurrences()) {
    auto schema = fabric.memoryConfigurationSchema(candidate);
    if (!schema)
      return schema.takeError();
    const std::uint64_t rowCount = schema->layout().operationRows.size();
    if (!ranges.emplace(candidate.id(), Range{firstContext, rowCount}).second)
      return invalid("memory occurrence appears more than once");
    if (rowCount > std::numeric_limits<std::uint64_t>::max() - firstContext)
      return invalid("request context inventory overflows uint64");
    firstContext += rowCount;
  }
  return PortableMemoryRequestContextIndex(std::move(ranges));
}

llvm::Expected<std::uint64_t> PortableMemoryRequestContextIndex::code(
    fabric::FabricMemoryOccurrenceRef memory,
    std::uint64_t operationRowOrdinal) const {
  const auto found = ranges_.find(memory.id());
  if (found == ranges_.end())
    return invalid("memory occurrence is not owned by the Fabric artifact");
  if (operationRowOrdinal >= found->second.count)
    return invalid("operation row ordinal is outside its Fabric domain");
  if (operationRowOrdinal >
      std::numeric_limits<std::uint64_t>::max() - found->second.first)
    return invalid("request context code overflows uint64");
  return found->second.first + operationRowOrdinal;
}

llvm::Expected<std::uint64_t> PortableMemoryRequestContextIndex::first(
    fabric::FabricMemoryOccurrenceRef memory) const {
  const auto found = ranges_.find(memory.id());
  if (found == ranges_.end())
    return invalid("memory occurrence is not owned by the Fabric artifact");
  return found->second.first;
}

} // namespace loom::hardware::rtl
