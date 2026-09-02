#include "Arbitration.h"
#include "Components.h"

#include "Common/InvocationDiagnosticLog.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/IR/MemoryCapabilityDomains.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/Identity/FabricMemoryConfiguration.h"
#include "Hardware/RTL/MemoryServiceTransport.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Support/BackedgeBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::hardware::rtl::hierarchy {
namespace {

using Role = ::dataflow::semantics::ServiceValueRole;

mlir::Value zero(mlir::OpBuilder &builder, mlir::Location location,
                 unsigned width) {
  return circt::hw::ConstantOp::create(builder, location,
                                       llvm::APInt(width, 0));
}

mlir::Value constant(mlir::OpBuilder &builder, mlir::Location location,
                     unsigned width, std::uint64_t value) {
  return circt::hw::ConstantOp::create(builder, location,
                                       llvm::APInt(width, value));
}

mlir::Value equals(mlir::OpBuilder &builder, mlir::Location location,
                   mlir::Value value, std::uint64_t expected) {
  const unsigned width =
      mlir::cast<mlir::IntegerType>(value.getType()).getWidth();
  return circt::comb::ICmpOp::create(
      builder, location, circt::comb::ICmpPredicate::eq, value,
      constant(builder, location, width, expected), true);
}

mlir::Value extract(mlir::OpBuilder &builder, mlir::Location location,
                    mlir::Value value, std::uint64_t offset,
                    std::uint64_t width) {
  if (width == 0)
    return zero(builder, location, 1);
  return circt::comb::ExtractOp::create(builder, location, value, offset,
                                        width);
}

mlir::Value adaptWidth(mlir::OpBuilder &builder, mlir::Location location,
                       mlir::Value value, unsigned width) {
  const unsigned sourceWidth =
      mlir::cast<mlir::IntegerType>(value.getType()).getWidth();
  if (sourceWidth == width)
    return value;
  if (sourceWidth > width)
    return circt::comb::ExtractOp::create(builder, location, value, 0, width);
  return circt::comb::ConcatOp::create(
      builder, location,
      llvm::ArrayRef<mlir::Value>{zero(builder, location, width - sourceWidth),
                                  value});
}

mlir::Value indexedValue(mlir::OpBuilder &builder, mlir::Location location,
                         mlir::Value index,
                         llvm::ArrayRef<mlir::Value> lowToHigh) {
  assert(!lowToHigh.empty() && "indexed domain must not be empty");
  if (lowToHigh.size() == 1)
    return lowToHigh.front();
  llvm::SmallVector<mlir::Value> highToLow(llvm::reverse(lowToHigh));
  mlir::Value array =
      circt::hw::ArrayCreateOp::create(builder, location, highToLow);
  return circt::hw::ArrayGetOp::create(builder, location, array, index);
}

mlir::Value oneHotOrdinal(mlir::OpBuilder &builder, mlir::Location location,
                          llvm::ArrayRef<mlir::Value> oneHot) {
  const unsigned width = indexWidth(oneHot.size());
  llvm::SmallVector<mlir::Value> highToLow;
  highToLow.reserve(width);
  for (unsigned bit = width; bit != 0; --bit) {
    llvm::SmallVector<mlir::Value> selected;
    for (std::size_t ordinal = 0; ordinal != oneHot.size(); ++ordinal)
      if ((ordinal >> (bit - 1)) & 1U)
        selected.push_back(oneHot[ordinal]);
    highToLow.push_back(orValues(builder, location, selected));
  }
  if (width == 1)
    return highToLow.front();
  return circt::comb::ConcatOp::create(builder, location, highToLow);
}

unsigned roleWidth(Role role, unsigned addressWidth, unsigned dataWidth,
                   unsigned maskWidth) {
  switch (role) {
  case Role::Address:
    return addressWidth;
  case Role::Data:
  case Role::Update:
  case Role::Expected:
  case Role::Desired:
  case Role::Old:
    return dataWidth;
  case Role::Mask:
  case Role::Success:
    return maskWidth;
  case Role::Payload:
  case Role::Control:
  case Role::Completion:
    return 0;
  }
  llvm_unreachable("unknown Canonical Service role");
}

struct AccessCase final {
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

struct MemoryMaterializationMetrics final {
  std::uint64_t beginOperations = 0;
  std::uint64_t rowTransportOperations = 0;
  std::uint64_t requestFormationOperations = 0;
  std::uint64_t arbitrationOperations = 0;
  std::uint64_t issueReadinessOperations = 0;
  std::uint64_t localStorageOperations = 0;
  std::uint64_t completionOperations = 0;
};

std::uint64_t blockOperationCount(mlir::OpBuilder &builder) {
  return static_cast<std::uint64_t>(
      std::distance(builder.getBlock()->begin(), builder.getBlock()->end()));
}

std::uint64_t saturatedProduct(std::initializer_list<std::uint64_t> values) {
  std::uint64_t result = 1;
  for (std::uint64_t value : values) {
    if (value != 0 &&
        result > std::numeric_limits<std::uint64_t>::max() / value)
      return std::numeric_limits<std::uint64_t>::max();
    result *= value;
  }
  return result;
}

void replaceAll(std::string &text, llvm::StringRef from, llvm::StringRef to) {
  if (from.empty())
    return;
  std::size_t position = 0;
  while ((position = text.find(from.str(), position)) != std::string::npos) {
    text.replace(position, from.size(), to.str());
    position += to.size();
  }
}

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

llvm::Expected<std::vector<AccessCase>>
deriveAccessCases(const fabric::FabricArtifactView &fabric,
                  fabric::FabricMemoryOccurrenceRef memory) {
  std::vector<AccessCase> result;
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
            AccessCase selected;
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

llvm::Expected<std::vector<AccessCase>>
deriveLocalServiceAccessCases(const fabric::FabricArtifactView &fabric,
                              fabric::FabricMemoryOccurrenceRef memory) {
  std::vector<AccessCase> result;
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
          AccessCase selected;
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

struct RowSignals final {
  mlir::Value active;
  mlir::Value physicalPort;
  mlir::Value capability;
  mlir::Value accessClass;
  mlir::Value elementWidth;
  mlir::Value laneCount;
  mlir::Value maskPair;
  mlir::Value addressLaneWidth;
  mlir::Value baseAddress;
  mlir::Value serviceTarget;
  std::vector<mlir::Value> sourcePresent;
  std::vector<mlir::Value> sourceInternal;
  std::vector<mlir::Value> sourceEndpoint;
  std::vector<mlir::Value> sourceConnection;
  std::vector<mlir::Value> sourceTag;
  std::vector<mlir::Value> destinationExternal;
  std::vector<mlir::Value> destinationEndpoint;
  std::vector<mlir::Value> destinationTag;
  std::vector<std::vector<mlir::Value>> destinationInternal;
};

RowSignals decodeRow(mlir::OpBuilder &builder, mlir::Location location,
                     mlir::Value field,
                     const fabric::FabricMemoryConfigurationLayout &layout,
                     const fabric::FabricMemoryOperationRowLayout &row,
                     mlir::Value memoryActive) {
  RowSignals result;
  result.active = andValues(
      builder, location,
      {memoryActive, selectedBit(builder, location, field, row.bitOffset)});
  result.physicalPort =
      extract(builder, location, field, row.physicalPortOffset,
              layout.physicalPortBitCount);
  result.capability = extract(builder, location, field, row.capabilityOffset,
                              layout.capabilityBitCount);
  result.accessClass = extract(builder, location, field, row.accessClassOffset,
                               layout.accessClassBitCount);
  result.elementWidth =
      extract(builder, location, field, row.elementWidthOffset, 64);
  result.laneCount = extract(builder, location, field, row.laneCountOffset, 64);
  result.maskPair = extract(builder, location, field, row.maskPairOffset,
                            layout.maskPairBitCount);
  result.addressLaneWidth =
      extract(builder, location, field, row.addressLaneWidthOffset, 32);
  result.baseAddress =
      extract(builder, location, field, row.baseAddressOffset, 64);
  result.serviceTarget =
      extract(builder, location, field, row.serviceTargetOffset,
              layout.serviceTargetBitCount);
  for (std::uint32_t role = 0; role != layout.roleCount; ++role) {
    const std::uint64_t source = row.roleSourceOffsets[role];
    result.sourcePresent.push_back(
        selectedBit(builder, location, field, source));
    result.sourceInternal.push_back(
        selectedBit(builder, location, field, source + 1));
    result.sourceEndpoint.push_back(extract(builder, location, field,
                                            source + 2,
                                            layout.transportEndpointBitCount));
    result.sourceConnection.push_back(extract(
        builder, location, field, source + 2 + layout.transportEndpointBitCount,
        layout.internalConnectionBitCount));
    result.sourceTag.push_back(extract(builder, location, field,
                                       source + 2 +
                                           layout.transportEndpointBitCount +
                                           layout.internalConnectionBitCount,
                                       layout.tagWidthBits));

    const std::uint64_t destination = row.roleDestinationOffsets[role];
    result.destinationExternal.push_back(
        selectedBit(builder, location, field, destination));
    result.destinationEndpoint.push_back(
        extract(builder, location, field, destination + 1,
                layout.transportEndpointBitCount));
    result.destinationTag.push_back(
        extract(builder, location, field,
                destination + 1 + layout.transportEndpointBitCount,
                layout.tagWidthBits));
    std::vector<mlir::Value> internal;
    const std::uint64_t connectionBase = destination + 1 +
                                         layout.transportEndpointBitCount +
                                         layout.tagWidthBits;
    internal.reserve(layout.internalConnectionCount);
    for (std::uint32_t connection = 0;
         connection != layout.internalConnectionCount; ++connection)
      internal.push_back(
          selectedBit(builder, location, field, connectionBase + connection));
    result.destinationInternal.push_back(std::move(internal));
  }
  return result;
}

mlir::Value accessCaseMatches(mlir::OpBuilder &builder, mlir::Location location,
                              const RowSignals &row, const AccessCase &access,
                              std::optional<std::uint32_t> spatialPort) {
  llvm::SmallVector<mlir::Value> terms{
      row.active, equals(builder, location, row.capability, access.capability),
      equals(builder, location, row.accessClass, access.accessClass),
      equals(builder, location, row.elementWidth, access.elementWidthBits)};
  if (spatialPort) {
    if (*spatialPort != access.physicalPort)
      return bitConstant(builder, location, false);
  } else {
    terms.push_back(
        equals(builder, location, row.physicalPort, access.physicalPort));
  }
  return andValues(builder, location, terms);
}

struct SourceRuntime final {
  mlir::Value valid;
  mlir::Value data;
};

struct MemoryOperandQueueRuntime final {
  circt::Backedge occupiedNext;
  circt::Backedge dataNext;
  circt::Backedge dequeue;
  mlir::Value occupied;
  mlir::Value data;
  mlir::Value available;
  mlir::Value enqueue;
  mlir::Value enqueueData;
};

struct ServiceRequestSignals final {
  mlir::Value kind;
  mlir::Value address;
  mlir::Value data;
  mlir::Value mask;
  mlir::Value activeLanesKind;
  mlir::Value accessForm;
  mlir::Value addressForm;
  mlir::Value elementWidth;
  mlir::Value laneCount;
  mlir::Value addressLaneWidth;
  mlir::Value baseAddress;
  mlir::Value context;
  mlir::Value valid;
};

ServiceRequestSignals zeroRequest(mlir::OpBuilder &builder,
                                  mlir::Location location,
                                  const PortableMemoryServiceLayout &layout) {
  return {zero(builder, location, 1),
          zero(builder, location, layout.addressWidthBits),
          zero(builder, location, layout.dataWidthBits),
          zero(builder, location, layout.maskWidthBits),
          zero(builder, location, 1),
          zero(builder, location, 2),
          zero(builder, location, 1),
          zero(builder, location, 64),
          zero(builder, location, 64),
          zero(builder, location, 32),
          zero(builder, location, 64),
          zero(builder, location, 64),
          bitConstant(builder, location, false)};
}

ServiceRequestSignals muxRequest(mlir::OpBuilder &builder,
                                 mlir::Location location, mlir::Value select,
                                 const ServiceRequestSignals &selected,
                                 const ServiceRequestSignals &fallback) {
  const auto mux = [&](mlir::Value lhs, mlir::Value rhs) {
    return mlir::Value(
        circt::comb::MuxOp::create(builder, location, select, lhs, rhs, true));
  };
  return {mux(selected.kind, fallback.kind),
          mux(selected.address, fallback.address),
          mux(selected.data, fallback.data),
          mux(selected.mask, fallback.mask),
          mux(selected.activeLanesKind, fallback.activeLanesKind),
          mux(selected.accessForm, fallback.accessForm),
          mux(selected.addressForm, fallback.addressForm),
          mux(selected.elementWidth, fallback.elementWidth),
          mux(selected.laneCount, fallback.laneCount),
          mux(selected.addressLaneWidth, fallback.addressLaneWidth),
          mux(selected.baseAddress, fallback.baseAddress),
          mux(selected.context, fallback.context),
          mux(selected.valid, fallback.valid)};
}

ServiceRequestSignals requestInput(circt::hw::HWModulePortAccessor &accessor,
                                   const MemoryServicePortPlan &ports) {
  return {accessor.getInput(ports.requestKind.getName()),
          accessor.getInput(ports.requestAddress.getName()),
          accessor.getInput(ports.requestData.getName()),
          accessor.getInput(ports.requestMask.getName()),
          accessor.getInput(ports.requestActiveLanesKind.getName()),
          accessor.getInput(ports.requestAccessForm.getName()),
          accessor.getInput(ports.requestAddressForm.getName()),
          accessor.getInput(ports.requestElementWidth.getName()),
          accessor.getInput(ports.requestLaneCount.getName()),
          accessor.getInput(ports.requestAddressLaneWidth.getName()),
          accessor.getInput(ports.requestBaseAddress.getName()),
          accessor.getInput(ports.requestContext.getName()),
          accessor.getInput(ports.requestValid.getName())};
}

void setRequestOutputs(circt::hw::HWModulePortAccessor &accessor,
                       const MemoryServicePortPlan &ports,
                       const ServiceRequestSignals &request) {
  accessor.setOutput(ports.requestKind.getName(), request.kind);
  accessor.setOutput(ports.requestAddress.getName(), request.address);
  accessor.setOutput(ports.requestData.getName(), request.data);
  accessor.setOutput(ports.requestMask.getName(), request.mask);
  accessor.setOutput(ports.requestActiveLanesKind.getName(),
                     request.activeLanesKind);
  accessor.setOutput(ports.requestAccessForm.getName(), request.accessForm);
  accessor.setOutput(ports.requestAddressForm.getName(), request.addressForm);
  accessor.setOutput(ports.requestElementWidth.getName(), request.elementWidth);
  accessor.setOutput(ports.requestLaneCount.getName(), request.laneCount);
  accessor.setOutput(ports.requestAddressLaneWidth.getName(),
                     request.addressLaneWidth);
  accessor.setOutput(ports.requestBaseAddress.getName(), request.baseAddress);
  accessor.setOutput(ports.requestContext.getName(), request.context);
  accessor.setOutput(ports.requestValid.getName(), request.valid);
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

mlir::Value selectedAccessForm(mlir::OpBuilder &builder,
                               mlir::Location location,
                               llvm::ArrayRef<AccessCase> accesses,
                               llvm::ArrayRef<mlir::Value> matches,
                               bool addressForm) {
  assert(accesses.size() == matches.size() &&
         "access projection must cover its exact domain");
  mlir::Value result = zero(builder, location, addressForm ? 1 : 2);
  for (auto [index, access] : llvm::enumerate(accesses)) {
    const std::uint64_t code = addressForm ? addressFormCode(access.addressForm)
                                           : accessFormCode(access.accessForm);
    result = circt::comb::MuxOp::create(
        builder, location, matches[index],
        constant(builder, location, addressForm ? 1 : 2, code), result, true);
  }
  return result;
}

mlir::Value selectedDynamicMask(mlir::OpBuilder &builder,
                                mlir::Location location, const RowSignals &row,
                                llvm::ArrayRef<AccessCase> accesses,
                                llvm::ArrayRef<mlir::Value> matches) {
  assert(accesses.size() == matches.size() &&
         "access projection must cover its exact domain");
  mlir::Value result = bitConstant(builder, location, false);
  for (auto [index, access] : llvm::enumerate(accesses))
    for (std::size_t ordinal = 0; ordinal != access.dynamicMasks.size();
         ++ordinal)
      if (access.dynamicMasks[ordinal])
        result = circt::comb::OrOp::create(
            builder, location, result,
            andValues(builder, location,
                      {matches[index],
                       equals(builder, location, row.maskPair, ordinal)}));
  return result;
}

SourceRuntime sourceForRole(
    mlir::OpBuilder &builder, mlir::Location location,
    circt::hw::HWModulePortAccessor &accessor, const RowSignals &row,
    std::uint32_t role, unsigned width, llvm::ArrayRef<EndpointPlan> endpoints,
    std::optional<SourceRuntime> queuedExternal, SourceRuntime queuedInternal,
    const fabric::FabricMemoryConfigurationLayout &layout) {
  mlir::Value valid =
      circt::comb::createOrFoldNot(builder, location, row.sourcePresent[role]);
  mlir::Value data = zero(builder, location, std::max(1U, width));
  if (queuedExternal) {
    mlir::Value selected =
        andValues(builder, location,
                  {row.sourcePresent[role],
                   circt::comb::createOrFoldNot(builder, location,
                                                row.sourceInternal[role])});
    valid = circt::comb::OrOp::create(
        builder, location, valid,
        andValues(builder, location, {selected, queuedExternal->valid}));
    if (width != 0)
      data = circt::comb::MuxOp::create(
          builder, location, selected,
          adaptWidth(builder, location, queuedExternal->data, width), data,
          true);
  } else {
    for (const EndpointPlan &endpoint : endpoints) {
      if (endpoint.direction != fabric::FabricPortDirection::Input)
        continue;
      mlir::Value selected =
          andValues(builder, location,
                    {row.sourcePresent[role],
                     circt::comb::createOrFoldNot(builder, location,
                                                  row.sourceInternal[role]),
                     equals(builder, location, row.sourceEndpoint[role],
                            endpoint.endpoint.ordinal)});
      if (layout.tagWidthBits != 0)
        selected = andValues(
            builder, location,
            {selected, circt::comb::ICmpOp::create(
                           builder, location, circt::comb::ICmpPredicate::eq,
                           row.sourceTag[role],
                           accessor.getInput(endpoint.tag->getName()), true)});
      valid = circt::comb::OrOp::create(
          builder, location, valid,
          andValues(builder, location,
                    {selected, accessor.getInput(endpoint.valid.getName())}));
      if (width != 0 && endpoint.data)
        data = circt::comb::MuxOp::create(
            builder, location, selected,
            adaptWidth(builder, location,
                       accessor.getInput(endpoint.data->getName()), width),
            data, true);
    }
  }
  mlir::Value selectedInternal = andValues(
      builder, location, {row.sourcePresent[role], row.sourceInternal[role]});
  valid = circt::comb::OrOp::create(
      builder, location, valid,
      andValues(builder, location, {selectedInternal, queuedInternal.valid}));
  if (width != 0)
    data = circt::comb::MuxOp::create(
        builder, location, selectedInternal,
        adaptWidth(builder, location, queuedInternal.data, width), data, true);
  return {valid, data};
}

struct AddressedByte final {
  mlir::Value active;
  mlir::Value address;
};

mlir::Value serviceAccessMatches(mlir::OpBuilder &builder,
                                 mlir::Location location,
                                 const ServiceRequestSignals &request,
                                 const AccessCase &access) {
  llvm::SmallVector<mlir::Value> terms{
      equals(builder, location, request.kind, access.read ? 0 : 1),
      equals(builder, location, request.accessForm,
             accessFormCode(access.accessForm)),
      equals(builder, location, request.addressForm,
             addressFormCode(access.addressForm)),
      equals(builder, location, request.elementWidth, access.elementWidthBits)};
  mlir::Value supportedWidth = bitConstant(builder, location, false);
  for (std::uint32_t width : access.addressWidths)
    supportedWidth = circt::comb::OrOp::create(
        builder, location, supportedWidth,
        equals(builder, location, request.addressLaneWidth, width));
  terms.push_back(supportedWidth);
  mlir::Value supportedLanes = bitConstant(builder, location, false);
  for (const ::fabric::UnsignedInterval interval : access.laneCounts) {
    mlir::Value inInterval =
        andValues(builder, location,
                  {circt::comb::ICmpOp::create(
                       builder, location, circt::comb::ICmpPredicate::uge,
                       request.laneCount,
                       constant(builder, location, 64, interval.lower), true),
                   circt::comb::ICmpOp::create(
                       builder, location, circt::comb::ICmpPredicate::ule,
                       request.laneCount,
                       constant(builder, location, 64, interval.upper), true)});
    supportedLanes = circt::comb::OrOp::create(builder, location,
                                               supportedLanes, inInterval);
  }
  terms.push_back(supportedLanes);
  const bool admitsAll = llvm::is_contained(access.dynamicMasks, false);
  const bool admitsBits = llvm::is_contained(access.dynamicMasks, true);
  terms.push_back(circt::comb::MuxOp::create(
      builder, location, request.activeLanesKind,
      bitConstant(builder, location, admitsBits),
      bitConstant(builder, location, admitsAll), true));
  return andValues(builder, location, terms);
}

AddressedByte serviceAddressByte(
    mlir::OpBuilder &builder, mlir::Location location,
    const ServiceRequestSignals &request, const AccessCase &access,
    mlir::Value accessSelected, std::uint64_t byte,
    const PortableMemoryServiceLayout &layout,
    const PortableMemoryAddressArithmetic &arithmetic) {
  const unsigned calculationWidth = arithmetic.calculationWidthBits;
  mlir::Value active = bitConstant(builder, location, false);
  mlir::Value address = zero(builder, location, calculationWidth);
  const std::uint64_t elementBytes = access.elementWidthBits / 8;
  const std::uint64_t lane = byte / elementBytes;
  const std::uint64_t byteWithinLane = byte % elementBytes;
  mlir::Value selected = andValues(
      builder, location,
      {request.valid, accessSelected,
       circt::comb::ICmpOp::create(
           builder, location, circt::comb::ICmpPredicate::ult,
           constant(builder, location, 64, lane), request.laneCount, true)});
  mlir::Value maskBit = lane < layout.maskWidthBits
                            ? extract(builder, location, request.mask, lane, 1)
                            : bitConstant(builder, location, false);
  selected = andValues(
      builder, location,
      {selected, circt::comb::MuxOp::create(
                     builder, location, request.activeLanesKind, maskBit,
                     bitConstant(builder, location, true), true)});

  for (std::uint32_t width : access.addressWidths) {
    const std::uint64_t laneOffset =
        access.accessForm == ::dataflow::semantics::MemoryAccessForm::Indexed
            ? lane * width
            : 0;
    if (laneOffset + width > layout.addressWidthBits)
      continue;
    mlir::Value laneAddress =
        extract(builder, location, request.address, laneOffset, width);
    laneAddress = adaptWidth(builder, location, laneAddress, calculationWidth);
    mlir::Value byteAddress = laneAddress;
    if (access.addressForm ==
        ::dataflow::semantics::MemoryAddressForm::RootRelative)
      byteAddress = circt::comb::AddOp::create(
          builder, location,
          adaptWidth(builder, location, request.baseAddress, calculationWidth),
          circt::comb::MulOp::create(
              builder, location, laneAddress,
              constant(builder, location, calculationWidth, elementBytes),
              true),
          true);
    const std::uint64_t byteOffset =
        access.accessForm == ::dataflow::semantics::MemoryAccessForm::Indexed
            ? byteWithinLane
            : byte;
    byteAddress = circt::comb::AddOp::create(
        builder, location, byteAddress,
        constant(builder, location, calculationWidth, byteOffset), true);
    mlir::Value widthSelected = andValues(
        builder, location,
        {selected, equals(builder, location, request.addressLaneWidth, width)});
    address = circt::comb::MuxOp::create(builder, location, widthSelected,
                                         byteAddress, address, true);
    active =
        circt::comb::OrOp::create(builder, location, active, widthSelected);
  }
  return {active, address};
}

struct LocalRequestProjection final {
  mlir::Value legal;
  std::vector<AddressedByte> bytes;
};

LocalRequestProjection projectLocalRequest(
    mlir::OpBuilder &builder, mlir::Location location,
    const ServiceRequestSignals &request, llvm::ArrayRef<AccessCase> accesses,
    std::uint64_t dataBytes, const PortableMemoryServiceLayout &layout,
    const PortableMemoryAddressArithmetic &arithmetic) {
  const unsigned calculationWidth = arithmetic.calculationWidthBits;
  mlir::Value supported = bitConstant(builder, location, false);
  mlir::Value outOfRange = bitConstant(builder, location, false);
  std::vector<AddressedByte> bytes(
      dataBytes, AddressedByte{bitConstant(builder, location, false),
                               zero(builder, location, calculationWidth)});
  for (const AccessCase &access : accesses) {
    mlir::Value accessSelected =
        serviceAccessMatches(builder, location, request, access);
    supported =
        circt::comb::OrOp::create(builder, location, supported, accessSelected);
    for (std::uint64_t byte = 0; byte != dataBytes; ++byte) {
      AddressedByte addressed =
          serviceAddressByte(builder, location, request, access,
                             accessSelected, byte, layout, arithmetic);
      bytes[byte].address = circt::comb::MuxOp::create(
          builder, location, addressed.active, addressed.address,
          bytes[byte].address, true);
      bytes[byte].active = circt::comb::OrOp::create(
          builder, location, bytes[byte].active, addressed.active);
      mlir::Value inRegion = bitConstant(builder, location, false);
      for (const ::fabric::UnsignedInterval region : access.storageRegions) {
        mlir::Value inSelectedRegion = circt::comb::ICmpOp::create(
            builder, location, circt::comb::ICmpPredicate::ule,
            addressed.address,
            constant(builder, location, calculationWidth, region.upper), true);
        if (region.lower != 0)
          inSelectedRegion = andValues(
              builder, location,
              {inSelectedRegion,
               circt::comb::ICmpOp::create(
                   builder, location, circt::comb::ICmpPredicate::uge,
                   addressed.address,
                   constant(builder, location, calculationWidth, region.lower),
                   true)});
        inRegion = circt::comb::OrOp::create(builder, location, inRegion,
                                             inSelectedRegion);
      }
      outOfRange = circt::comb::OrOp::create(
          builder, location, outOfRange,
          andValues(
              builder, location,
              {accessSelected, addressed.active,
               circt::comb::createOrFoldNot(builder, location, inRegion)}));
    }
  }
  return {andValues(builder, location,
                    {supported, circt::comb::createOrFoldNot(builder, location,
                                                             outOfRange)}),
          std::move(bytes)};
}

/// The byte address of a service request's first lane, evaluated in the
/// portable calculation width. `exact` holds when the lane width lies within
/// the byte-address domain and no bit of the complete expression lies at or
/// above that domain. Provider decode consumes the narrowed address only
/// under `exact`, so a wrapped address never selects a Range or Prefix row.
struct FirstServiceByteAddress final {
  mlir::Value byteAddress;
  mlir::Value exact;
};

FirstServiceByteAddress
firstServiceByteAddress(mlir::OpBuilder &builder, mlir::Location location,
                        const ServiceRequestSignals &request,
                        const PortableMemoryServiceLayout &layout,
                        const PortableMemoryAddressArithmetic &arithmetic) {
  const unsigned calculationWidth = arithmetic.calculationWidthBits;
  const unsigned byteAddressWidth = arithmetic.byteAddressWidthBits;
  // The carrier packs Indexed lanes side by side; the first lane is exactly
  // the low `addressLaneWidth` bits of the carrier.
  mlir::Value laneCarrier = adaptWidth(
      builder, location,
      extract(builder, location, request.address, 0,
              std::min<std::uint32_t>(byteAddressWidth,
                                      layout.addressWidthBits)),
      calculationWidth);
  mlir::Value one = constant(builder, location, calculationWidth, 1);
  mlir::Value laneMask = circt::comb::SubOp::create(
      builder, location,
      circt::comb::ShlOp::create(
          builder, location, one,
          adaptWidth(builder, location, request.addressLaneWidth,
                     calculationWidth),
          true),
      one, true);
  mlir::Value first =
      circt::comb::AndOp::create(builder, location, laneCarrier, laneMask, true);
  mlir::Value elementBytes = adaptWidth(
      builder, location,
      circt::comb::ShrUOp::create(
          builder, location, request.elementWidth,
          constant(builder, location, portableMemoryElementWidthFieldWidth, 3),
          true),
      calculationWidth);
  mlir::Value relative = circt::comb::AddOp::create(
      builder, location,
      adaptWidth(builder, location, request.baseAddress, calculationWidth),
      circt::comb::MulOp::create(builder, location, first, elementBytes, true),
      true);
  mlir::Value byteAddress = circt::comb::MuxOp::create(
      builder, location, request.addressForm, first, relative, true);
  mlir::Value overflow = circt::comb::ICmpOp::create(
      builder, location, circt::comb::ICmpPredicate::ne,
      extract(builder, location, byteAddress, byteAddressWidth,
              calculationWidth - byteAddressWidth),
      zero(builder, location, calculationWidth - byteAddressWidth), true);
  mlir::Value laneWidthSupported = circt::comb::ICmpOp::create(
      builder, location, circt::comb::ICmpPredicate::ule,
      request.addressLaneWidth,
      constant(builder, location, portableMemoryAddressLaneWidthFieldWidth,
               byteAddressWidth),
      true);
  return {extract(builder, location, byteAddress, 0, byteAddressWidth),
          andValues(builder, location,
                    {circt::comb::createOrFoldNot(builder, location, overflow),
                     laneWidthSupported})};
}

std::string memoryName(fabric::FabricMemoryOccurrenceRef memory) {
  return "loom_memory_" + std::to_string(memory.id());
}

llvm::Expected<MemoryModule>
buildMemoryModule(mlir::OpBuilder &builder, mlir::Location location,
                  fabric::SpatialCoreOccurrenceRef spatialCore,
                  const fabric::FabricArtifactView &fabric,
                  const ConfigurationABI &configurationAbi,
                  const ConfigurationTransportLayout &transportLayout,
                  const ClockResetPlan &clockReset,
                  const PortableMemoryServiceLayout &memoryServiceLayout,
                  fabric::FabricMemoryOccurrenceRef memory,
                  llvm::StringRef materializationKey) {
  auto endpoints = deriveEndpointPlans(
      builder, fabric, fabric::FabricTransportEndpointOwnerRef::of(memory));
  if (!endpoints)
    return endpoints.takeError();
  auto memoryEndpoints = deriveMemoryEndpointPortPlans(builder, fabric, memory,
                                                       memoryServiceLayout);
  if (!memoryEndpoints)
    return memoryEndpoints.takeError();
  auto schema = fabric.memoryConfigurationSchema(memory);
  if (!schema)
    return schema.takeError();
  const auto &layout = schema->layout();
  auto decoder = prepareFieldDecoder(spatialCore, schema->field(),
                                     configurationAbi, transportLayout);
  if (!decoder)
    return decoder.takeError();
  if (decoder->encodedBitCount != layout.carrierBitCount)
    return invalid("memory carrier width differs from its typed schema");
  const auto *connectivity = fabric.memoryConnectivity(memory);
  if (!connectivity)
    return invalid("memory occurrence has no connectivity contract");
  for (const ::fabric::MemorySubordinateDispatchDeclaration &subordinate :
       connectivity->subordinateEndpoints())
    if (llvm::is_contained(subordinate.matchFields,
                           ::fabric::MemoryProviderMatchField::AddressSpace) ||
        llvm::is_contained(subordinate.matchFields,
                           ::fabric::MemoryProviderMatchField::Context))
      return unsupported(
          "portable memory profile has no address-space or context "
          "projection");
  const bool hasLocalService = fabric.declaresLocalMemoryService(memory);
  const std::uint64_t capacity = fabric.localMemoryServiceCapacityBytes(memory);
  if (hasLocalService && capacity == 0)
    return invalid("local memory has no positive storage capacity");
  if (capacity > std::numeric_limits<std::uint32_t>::max())
    return unsupported("portable local memory capacity exceeds 32-bit storage");

  const auto arithmetic =
      derivePortableMemoryAddressArithmetic(memoryServiceLayout);
  if (!arithmetic)
    return unsupported(
        "portable memory address lane exceeds the byte-address domain");

  auto accessCases = deriveAccessCases(fabric, memory);
  if (!accessCases)
    return accessCases.takeError();
  auto serviceAccessCases = deriveLocalServiceAccessCases(fabric, memory);
  if (!serviceAccessCases)
    return serviceAccessCases.takeError();

  std::vector<const MemoryEndpointPortPlan *> managerEndpoints;
  std::vector<const MemoryEndpointPortPlan *> subordinateEndpoints;
  for (const MemoryEndpointPortPlan &endpoint : *memoryEndpoints) {
    if (endpoint.ports.role == fabric::FabricMemoryEndpointRole::Manager)
      managerEndpoints.push_back(&endpoint);
    else
      subordinateEndpoints.push_back(&endpoint);
  }
  if (managerEndpoints.size() != layout.managerEndpointCount ||
      subordinateEndpoints.size() !=
          connectivity->subordinateEndpoints().size())
    return invalid("memory service endpoint inventories disagree");

  unsigned addressWidth = 1;
  unsigned dataWidth = 1;
  unsigned maskWidth = 1;
  for (const AccessCase &access : *accessCases) {
    const auto *port = fabric.memoryOperationPort(
        fabric::FabricMemoryOperationPortRef{memory, access.physicalPort});
    const auto &capability = port->capabilityAlternatives()[access.capability];
    for (const ::fabric::MemoryRoleEndpointBindingRecord &binding :
         capability.roleToEndpoint) {
      const auto endpoint =
          llvm::find_if(*endpoints, [&](const EndpointPlan &e) {
            return e.endpoint.ordinal == binding.endpointOrdinal;
          });
      if (endpoint == endpoints->end())
        return invalid("memory capability role endpoint does not resolve");
      const unsigned width = std::max(1U, endpoint->dataPath.payloadWidthBits);
      switch (binding.role) {
      case Role::Address:
        addressWidth = std::max(addressWidth, width);
        break;
      case Role::Mask:
      case Role::Success:
        maskWidth = std::max(maskWidth, width);
        break;
      case Role::Data:
      case Role::Update:
      case Role::Expected:
      case Role::Desired:
      case Role::Old:
        dataWidth = std::max(dataWidth, width);
        break;
      default:
        break;
      }
    }
  }
  addressWidth = std::max(addressWidth, memoryServiceLayout.addressWidthBits);
  dataWidth = std::max(dataWidth, memoryServiceLayout.dataWidthBits);
  maskWidth = std::max(maskWidth, memoryServiceLayout.maskWidthBits);
  const std::uint64_t dataBytes = (dataWidth + 7) / 8;
  const std::uint64_t rowCount = layout.operationRows.size();
  const std::uint64_t roleCount = layout.roleCount;
  const std::uint64_t connectionCount = layout.internalConnectionCount;
  const std::uint64_t inputEndpointCount =
      llvm::count_if(*endpoints, [](const EndpointPlan &endpoint) {
        return endpoint.direction == fabric::FabricPortDirection::Input;
      });
  const std::uint64_t outputEndpointCount =
      endpoints->size() - inputEndpointCount;
  const std::uint64_t semanticObligationCount = saturatedProduct(
      {rowCount, roleCount, outputEndpointCount + connectionCount});
  const std::uint64_t obligationCount =
      saturatedProduct({roleCount, outputEndpointCount + connectionCount});
  MemoryMaterializationMetrics metrics;

  llvm::SmallVector<circt::hw::PortInfo, 32> inputs;
  llvm::SmallVector<circt::hw::PortInfo, 32> outputs;
  auto configuration = deriveConfigurationBundlePlan(
      llvm::ArrayRef<FieldDecoderPlan>(&*decoder, 1));
  if (!configuration)
    return configuration.takeError();
  const ConfigurationBundlePlan emptyConfiguration;
  appendClockResetAndConfigurationPorts(builder, emptyConfiguration, inputs);
  inputs.push_back(circt::hw::PortInfo{
      {builder.getStringAttr(configurationValuePortName),
       builder.getIntegerType(static_cast<unsigned>(decoder->encodedBitCount)),
       circt::hw::ModulePort::Direction::Input}});
  inputs.push_back(
      {{builder.getStringAttr("request_context_base"), builder.getI64Type(),
        circt::hw::ModulePort::Direction::Input}});
  for (const EndpointPlan &endpoint : *endpoints)
    appendEndpointPorts(inputs, outputs, endpoint);
  for (const MemoryEndpointPortPlan &endpoint : *memoryEndpoints)
    appendMemoryServicePorts(inputs, outputs, endpoint.ports);

  std::optional<std::string> materializationError;
  auto module = circt::hw::HWModuleOp::create(
      builder, location, builder.getStringAttr(memoryName(memory)),
      circt::hw::ModulePortInfo(inputs, outputs),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        metrics.beginOperations = blockOperationCount(bodyBuilder);
        circt::BackedgeBuilder backedges(bodyBuilder, location);
        mlir::Value requestContextBase =
            accessor.getInput("request_context_base");
        mlir::Value field = accessor.getInput(configurationValuePortName);
        mlir::Value memoryActive = selectedBit(bodyBuilder, location, field, 0);
        std::vector<RowSignals> rows;
        rows.reserve(layout.operationRows.size());
        for (const auto &row : layout.operationRows)
          rows.push_back(decodeRow(bodyBuilder, location, field, layout, row,
                                   memoryActive));

        const std::uint32_t rowCount = rows.size();
        const std::uint32_t subordinateCount = subordinateEndpoints.size();
        const std::uint32_t requesterCount = rowCount + subordinateCount;
        const unsigned rowCursorWidth = indexWidth(rowCount);
        const unsigned requesterCursorWidth = indexWidth(requesterCount);
        std::vector<circt::Backedge> occupiedNext(rowCount);
        std::vector<mlir::Value> occupied(rowCount);
        std::vector<circt::Backedge> completedNext(rowCount);
        std::vector<mlir::Value> completed(rowCount);
        std::vector<circt::Backedge> resultDataNext(rowCount);
        std::vector<mlir::Value> resultData(rowCount);
        for (std::uint32_t row = 0; row != rowCount; ++row) {
          occupiedNext[row] = backedges.get(bodyBuilder.getI1Type());
          occupied[row] = createRegister(
              bodyBuilder, location, occupiedNext[row],
              accessor.getInput("clock"), accessor.getInput("reset"),
              llvm::APInt(1, 0), "result_occupied_" + std::to_string(row),
              clockReset.asynchronousReset);
          completedNext[row] = backedges.get(bodyBuilder.getI1Type());
          completed[row] = createRegister(
              bodyBuilder, location, completedNext[row],
              accessor.getInput("clock"), accessor.getInput("reset"),
              llvm::APInt(1, 0), "result_completed_" + std::to_string(row),
              clockReset.asynchronousReset);
          resultDataNext[row] =
              backedges.get(bodyBuilder.getIntegerType(dataWidth));
          resultData[row] = createRegister(
              bodyBuilder, location, resultDataNext[row],
              accessor.getInput("clock"), accessor.getInput("reset"),
              llvm::APInt(dataWidth, 0), "result_data_" + std::to_string(row),
              clockReset.asynchronousReset);
        }

        std::vector<std::vector<MemoryOperandQueueRuntime>> operandQueues(
            rowCount);
        std::map<std::uint64_t, std::vector<mlir::Value>> inputReadyTerms;
        for (std::uint32_t row = 0; row != rowCount; ++row) {
          operandQueues[row].reserve(layout.roleCount);
          for (std::uint32_t role = 0; role != layout.roleCount; ++role) {
            const unsigned width =
                std::max(1U, roleWidth(static_cast<Role>(role), addressWidth,
                                       dataWidth, maskWidth));
            MemoryOperandQueueRuntime queue;
            queue.occupiedNext = backedges.get(bodyBuilder.getI1Type());
            queue.dataNext = backedges.get(bodyBuilder.getIntegerType(width));
            queue.dequeue = backedges.get(bodyBuilder.getI1Type());
            queue.occupied =
                createRegister(bodyBuilder, location, queue.occupiedNext,
                               accessor.getInput("clock"),
                               accessor.getInput("reset"), llvm::APInt(1, 0),
                               "operand_occupied_" + std::to_string(row) + "_" +
                                   std::to_string(role),
                               clockReset.asynchronousReset);
            queue.data = createRegister(bodyBuilder, location, queue.dataNext,
                                        accessor.getInput("clock"),
                                        accessor.getInput("reset"),
                                        llvm::APInt(width, 0),
                                        "operand_data_" + std::to_string(row) +
                                            "_" + std::to_string(role),
                                        clockReset.asynchronousReset);
            queue.available = circt::comb::createOrFoldNot(
                bodyBuilder, location, queue.occupied);
            queue.enqueue = bitConstant(bodyBuilder, location, false);
            queue.enqueueData = zero(bodyBuilder, location, width);
            if (layout.schedule == ::fabric::Schedule::Temporal)
              for (const EndpointPlan &endpoint : *endpoints) {
                if (endpoint.direction != fabric::FabricPortDirection::Input)
                  continue;
                mlir::Value selected = andValues(
                    bodyBuilder, location,
                    {rows[row].active, rows[row].sourcePresent[role],
                     circt::comb::createOrFoldNot(
                         bodyBuilder, location, rows[row].sourceInternal[role]),
                     equals(bodyBuilder, location,
                            rows[row].sourceEndpoint[role],
                            endpoint.endpoint.ordinal)});
                selected = andValues(
                    bodyBuilder, location,
                    {selected,
                     circt::comb::ICmpOp::create(
                         bodyBuilder, location, circt::comb::ICmpPredicate::eq,
                         rows[row].sourceTag[role],
                         accessor.getInput(endpoint.tag->getName()), true)});
                inputReadyTerms[endpoint.endpoint.ordinal].push_back(andValues(
                    bodyBuilder, location, {selected, queue.available}));
                mlir::Value enqueue =
                    andValues(bodyBuilder, location,
                              {selected, queue.available,
                               accessor.getInput(endpoint.valid.getName())});
                queue.enqueue = circt::comb::OrOp::create(
                    bodyBuilder, location, queue.enqueue, enqueue);
                if (endpoint.data)
                  queue.enqueueData = circt::comb::MuxOp::create(
                      bodyBuilder, location, enqueue,
                      adaptWidth(bodyBuilder, location,
                                 accessor.getInput(endpoint.data->getName()),
                                 width),
                      queue.enqueueData, true);
              }
            operandQueues[row].push_back(std::move(queue));
          }
        }

        std::vector<std::vector<mlir::Value>> internalQueueMatches(
            layout.internalConnectionCount);
        std::vector<mlir::Value> internalQueueReady;
        internalQueueReady.reserve(layout.internalConnectionCount);
        for (std::uint32_t connection = 0;
             connection != layout.internalConnectionCount; ++connection) {
          llvm::SmallVector<mlir::Value> readyTerms;
          mlir::Value anyMatch = bitConstant(bodyBuilder, location, false);
          internalQueueMatches[connection].reserve(
              static_cast<std::size_t>(rowCount) * layout.roleCount);
          readyTerms.reserve(static_cast<std::size_t>(rowCount) *
                             layout.roleCount);
          for (std::uint32_t row = 0; row != rowCount; ++row)
            for (std::uint32_t role = 0; role != layout.roleCount; ++role) {
              mlir::Value matches = andValues(
                  bodyBuilder, location,
                  {rows[row].active, rows[row].sourcePresent[role],
                   rows[row].sourceInternal[role],
                   equals(bodyBuilder, location,
                          rows[row].sourceConnection[role], connection)});
              internalQueueMatches[connection].push_back(matches);
              anyMatch = circt::comb::OrOp::create(bodyBuilder, location,
                                                   anyMatch, matches);
              readyTerms.push_back(circt::comb::OrOp::create(
                  bodyBuilder, location,
                  circt::comb::createOrFoldNot(bodyBuilder, location, matches),
                  operandQueues[row][role].available));
            }
          internalQueueReady.push_back(andValues(
              bodyBuilder, location,
              {anyMatch, andValues(bodyBuilder, location, readyTerms)}));
        }

        struct PublishedObligation final {
          std::uint32_t role = 0;
          const EndpointPlan *endpoint = nullptr;
          std::optional<std::uint32_t> internalConnection;
          mlir::Value selected;
        };
        std::vector<PublishedObligation> obligations;
        std::vector<mlir::Value> released(
            rowCount, bitConstant(bodyBuilder, location, false));
        std::vector<mlir::Value> resultSelected(
            rowCount, bitConstant(bodyBuilder, location, false));
        mlir::Value selectedResultData = zero(bodyBuilder, location, dataWidth);
        std::vector<mlir::Value> selectedDestinationTag(
            layout.roleCount, zero(bodyBuilder, location, layout.tagWidthBits));
        std::optional<AtomicResultTupleSignals> publication;
        if (rowCount != 0) {
          circt::Backedge resultCursorNext =
              backedges.get(bodyBuilder.getIntegerType(rowCursorWidth));
          mlir::Value resultCursor = createRegister(
              bodyBuilder, location, resultCursorNext,
              accessor.getInput("clock"), accessor.getInput("reset"),
              llvm::APInt(rowCursorWidth, 0), "result_cursor",
              clockReset.asynchronousReset);
          resultSelected = roundRobinSelection(bodyBuilder, location, completed,
                                               resultCursor);
          mlir::Value selectedRow =
              oneHotOrdinal(bodyBuilder, location, resultSelected);
          mlir::Value anySelected =
              orValues(bodyBuilder, location, resultSelected);
          selectedResultData =
              indexedValue(bodyBuilder, location, selectedRow, resultData);
          std::vector<mlir::Value> selectedDestinationExternal(
              layout.roleCount);
          std::vector<mlir::Value> selectedDestinationEndpoint(
              layout.roleCount);
          std::vector<std::vector<mlir::Value>> selectedDestinationInternal(
              layout.roleCount,
              std::vector<mlir::Value>(layout.internalConnectionCount));
          for (std::uint32_t role = 0; role != layout.roleCount; ++role) {
            llvm::SmallVector<mlir::Value> external;
            llvm::SmallVector<mlir::Value> endpoint;
            llvm::SmallVector<mlir::Value> tag;
            external.reserve(rowCount);
            endpoint.reserve(rowCount);
            tag.reserve(rowCount);
            for (std::uint32_t row = 0; row != rowCount; ++row) {
              external.push_back(rows[row].destinationExternal[role]);
              endpoint.push_back(rows[row].destinationEndpoint[role]);
              tag.push_back(rows[row].destinationTag[role]);
            }
            selectedDestinationExternal[role] =
                indexedValue(bodyBuilder, location, selectedRow, external);
            selectedDestinationEndpoint[role] =
                indexedValue(bodyBuilder, location, selectedRow, endpoint);
            selectedDestinationTag[role] =
                indexedValue(bodyBuilder, location, selectedRow, tag);
            for (std::uint32_t connection = 0;
                 connection != layout.internalConnectionCount; ++connection) {
              llvm::SmallVector<mlir::Value> internal;
              internal.reserve(rowCount);
              for (std::uint32_t row = 0; row != rowCount; ++row)
                internal.push_back(
                    rows[row].destinationInternal[role][connection]);
              selectedDestinationInternal[role][connection] =
                  indexedValue(bodyBuilder, location, selectedRow, internal);
            }
          }
          llvm::SmallVector<mlir::Value> heldValids;
          llvm::SmallVector<mlir::Value> downstreamReady;
          for (std::uint32_t role = 0; role != layout.roleCount; ++role)
            for (const EndpointPlan &endpoint : *endpoints) {
              if (endpoint.direction != fabric::FabricPortDirection::Output)
                continue;
              mlir::Value selected =
                  andValues(bodyBuilder, location,
                            {anySelected, selectedDestinationExternal[role],
                             equals(bodyBuilder, location,
                                    selectedDestinationEndpoint[role],
                                    endpoint.endpoint.ordinal)});
              obligations.push_back({role, &endpoint, std::nullopt, selected});
              heldValids.push_back(selected);
              downstreamReady.push_back(
                  accessor.getInput(endpoint.ready.getName()));
            }
          for (std::uint32_t role = 0; role != layout.roleCount; ++role)
            for (std::uint32_t connection = 0;
                 connection != layout.internalConnectionCount; ++connection) {
              mlir::Value selected = andValues(
                  bodyBuilder, location,
                  {anySelected, selectedDestinationInternal[role][connection]});
              obligations.push_back({role, nullptr, connection, selected});
              heldValids.push_back(selected);
              downstreamReady.push_back(internalQueueReady[connection]);
            }
          auto derived =
              heldValids.empty()
                  ? llvm::Expected<AtomicResultTupleSignals>(
                        invalid("memory result has no transport destination "
                                "domain"))
                  : deriveAtomicResultTupleSignals(bodyBuilder, location,
                                                   heldValids, downstreamReady);
          if (!derived) {
            materializationError = llvm::toString(derived.takeError());
            backedges.abandon();
            return;
          }
          publication = std::move(*derived);
          for (std::uint32_t row = 0; row != rowCount; ++row)
            released[row] =
                andValues(bodyBuilder, location,
                          {resultSelected[row], publication->released});
          resultCursorNext.setValue(
              nextCursor(bodyBuilder, location, resultCursor, released));
        }

        for (const EndpointPlan &endpoint : *endpoints) {
          if (endpoint.direction != fabric::FabricPortDirection::Output)
            continue;
          mlir::Value valid = bitConstant(bodyBuilder, location, false);
          mlir::Value data = endpoint.data
                                 ? zero(bodyBuilder, location,
                                        endpoint.dataPath.payloadWidthBits)
                                 : mlir::Value{};
          mlir::Value tag = endpoint.tag ? zero(bodyBuilder, location,
                                                endpoint.dataPath.tagWidthBits)
                                         : mlir::Value{};
          for (auto [ordinal, obligation] : llvm::enumerate(obligations)) {
            if (!obligation.endpoint || obligation.endpoint != &endpoint)
              continue;
            valid = circt::comb::OrOp::create(
                bodyBuilder, location, valid,
                publication->publishedValids[ordinal]);
            if (data)
              data = circt::comb::MuxOp::create(
                  bodyBuilder, location, obligation.selected,
                  adaptWidth(bodyBuilder, location, selectedResultData,
                             endpoint.dataPath.payloadWidthBits),
                  data, true);
            if (tag)
              tag = circt::comb::MuxOp::create(
                  bodyBuilder, location, obligation.selected,
                  selectedDestinationTag[obligation.role], tag, true);
          }
          if (data)
            accessor.setOutput(endpoint.data->getName(), data);
          if (tag)
            accessor.setOutput(endpoint.tag->getName(), tag);
          accessor.setOutput(endpoint.valid.getName(), valid);
        }
        std::vector<mlir::Value> internalPublishedValid(
            layout.internalConnectionCount,
            bitConstant(bodyBuilder, location, false));
        std::vector<mlir::Value> internalPublishedData(
            layout.internalConnectionCount,
            zero(bodyBuilder, location, dataWidth));
        if (publication)
          for (auto [ordinal, obligation] : llvm::enumerate(obligations)) {
            if (!obligation.internalConnection)
              continue;
            const std::uint32_t connection = *obligation.internalConnection;
            internalPublishedValid[connection] = circt::comb::OrOp::create(
                bodyBuilder, location, internalPublishedValid[connection],
                publication->publishedValids[ordinal]);
            internalPublishedData[connection] = circt::comb::MuxOp::create(
                bodyBuilder, location, publication->publishedValids[ordinal],
                selectedResultData, internalPublishedData[connection], true);
          }
        for (std::uint32_t connection = 0;
             connection != layout.internalConnectionCount; ++connection) {
          std::size_t matchOrdinal = 0;
          for (std::uint32_t row = 0; row != rowCount; ++row)
            for (std::uint32_t role = 0; role != layout.roleCount;
                 ++role, ++matchOrdinal) {
              MemoryOperandQueueRuntime &queue = operandQueues[row][role];
              mlir::Value enqueue =
                  andValues(bodyBuilder, location,
                            {internalQueueMatches[connection][matchOrdinal],
                             internalPublishedValid[connection]});
              queue.enqueue = circt::comb::OrOp::create(bodyBuilder, location,
                                                        queue.enqueue, enqueue);
              queue.enqueueData = circt::comb::MuxOp::create(
                  bodyBuilder, location, enqueue,
                  adaptWidth(bodyBuilder, location,
                             internalPublishedData[connection],
                             queue.data.getType().getIntOrFloatBitWidth()),
                  queue.enqueueData, true);
            }
        }

        metrics.rowTransportOperations =
            blockOperationCount(bodyBuilder) - metrics.beginOperations;
        std::vector<circt::Backedge> subordinateBusyNext(subordinateCount);
        std::vector<mlir::Value> subordinateBusy(subordinateCount);
        std::vector<circt::Backedge> subordinateCompletedNext(subordinateCount);
        std::vector<mlir::Value> subordinateCompleted(subordinateCount);
        std::vector<circt::Backedge> subordinateDataNext(subordinateCount);
        std::vector<mlir::Value> subordinateData(subordinateCount);
        std::vector<mlir::Value> subordinateReleased(
            subordinateCount, bitConstant(bodyBuilder, location, false));
        for (std::uint32_t ordinal = 0; ordinal != subordinateCount;
             ++ordinal) {
          const auto &ports = subordinateEndpoints[ordinal]->ports;
          subordinateBusyNext[ordinal] = backedges.get(bodyBuilder.getI1Type());
          subordinateBusy[ordinal] = createRegister(
              bodyBuilder, location, subordinateBusyNext[ordinal],
              accessor.getInput("clock"), accessor.getInput("reset"),
              llvm::APInt(1, 0), "subordinate_busy_" + std::to_string(ordinal),
              clockReset.asynchronousReset);
          subordinateCompletedNext[ordinal] =
              backedges.get(bodyBuilder.getI1Type());
          subordinateCompleted[ordinal] = createRegister(
              bodyBuilder, location, subordinateCompletedNext[ordinal],
              accessor.getInput("clock"), accessor.getInput("reset"),
              llvm::APInt(1, 0),
              "subordinate_completed_" + std::to_string(ordinal),
              clockReset.asynchronousReset);
          subordinateDataNext[ordinal] =
              backedges.get(bodyBuilder.getIntegerType(dataWidth));
          subordinateData[ordinal] = createRegister(
              bodyBuilder, location, subordinateDataNext[ordinal],
              accessor.getInput("clock"), accessor.getInput("reset"),
              llvm::APInt(dataWidth, 0),
              "subordinate_data_" + std::to_string(ordinal),
              clockReset.asynchronousReset);
          accessor.setOutput(ports.responseData.getName(),
                             subordinateData[ordinal]);
          accessor.setOutput(ports.responseValid.getName(),
                             subordinateCompleted[ordinal]);
          subordinateReleased[ordinal] =
              andValues(bodyBuilder, location,
                        {subordinateCompleted[ordinal],
                         accessor.getInput(ports.responseReady.getName())});
        }

        std::vector<std::vector<SourceRuntime>> sources(rowCount);
        std::vector<mlir::Value> rowReads(rowCount);
        std::vector<mlir::Value> rowWrites(rowCount);
        std::vector<ServiceRequestSignals> requests;
        requests.reserve(requesterCount);
        std::vector<mlir::Value> targetValid;
        targetValid.reserve(requesterCount);
        std::vector<mlir::Value> targetCode;
        targetCode.reserve(requesterCount);
        for (std::uint32_t row = 0; row != rowCount; ++row) {
          sources[row].reserve(layout.roleCount);
          llvm::SmallVector<mlir::Value> sourceValids;
          for (std::uint32_t role = 0; role != layout.roleCount; ++role) {
            const unsigned width = roleWidth(
                static_cast<Role>(role), addressWidth, dataWidth, maskWidth);
            std::optional<SourceRuntime> queuedExternal;
            if (layout.schedule == ::fabric::Schedule::Temporal)
              queuedExternal = SourceRuntime{operandQueues[row][role].occupied,
                                             operandQueues[row][role].data};
            sources[row].push_back(
                sourceForRole(bodyBuilder, location, accessor, rows[row], role,
                              width, *endpoints, queuedExternal,
                              {operandQueues[row][role].occupied,
                               operandQueues[row][role].data},
                              layout));
            sourceValids.push_back(sources[row].back().valid);
          }
          rowReads[row] = bitConstant(bodyBuilder, location, false);
          rowWrites[row] = bitConstant(bodyBuilder, location, false);
          const std::optional<std::uint32_t> spatialPort =
              layout.schedule == ::fabric::Schedule::Spatial
                  ? std::optional<std::uint32_t>(row)
                  : std::nullopt;
          std::vector<mlir::Value> rowAccessMatches;
          rowAccessMatches.reserve(accessCases->size());
          for (const AccessCase &access : *accessCases) {
            mlir::Value matches = accessCaseMatches(
                bodyBuilder, location, rows[row], access, spatialPort);
            rowAccessMatches.push_back(matches);
            mlir::Value &kind = access.read ? rowReads[row] : rowWrites[row];
            kind =
                circt::comb::OrOp::create(bodyBuilder, location, kind, matches);
          }
          mlir::Value available = circt::comb::OrOp::create(
              bodyBuilder, location,
              circt::comb::createOrFoldNot(bodyBuilder, location,
                                           occupied[row]),
              released[row]);
          mlir::Value requestValid = andValues(
              bodyBuilder, location,
              {rows[row].active, available,
               orValues(bodyBuilder, location, {rowReads[row], rowWrites[row]}),
               andValues(bodyBuilder, location, sourceValids)});
          requests.push_back(
              {rowWrites[row],
               adaptWidth(
                   bodyBuilder, location,
                   sources[row][static_cast<unsigned>(Role::Address)].data,
                   memoryServiceLayout.addressWidthBits),
               adaptWidth(bodyBuilder, location,
                          sources[row][static_cast<unsigned>(Role::Data)].data,
                          memoryServiceLayout.dataWidthBits),
               adaptWidth(bodyBuilder, location,
                          sources[row][static_cast<unsigned>(Role::Mask)].data,
                          memoryServiceLayout.maskWidthBits),
               selectedDynamicMask(bodyBuilder, location, rows[row],
                                   *accessCases, rowAccessMatches),
               selectedAccessForm(bodyBuilder, location, *accessCases,
                                  rowAccessMatches, false),
               selectedAccessForm(bodyBuilder, location, *accessCases,
                                  rowAccessMatches, true),
               rows[row].elementWidth, rows[row].laneCount,
               rows[row].addressLaneWidth, rows[row].baseAddress,
               circt::comb::AddOp::create(
                   bodyBuilder, location, requestContextBase,
                   constant(bodyBuilder, location, 64, row), true),
               requestValid});
          targetValid.push_back(bitConstant(bodyBuilder, location, true));
          targetCode.push_back(rows[row].serviceTarget);
        }

        for (std::uint32_t ordinal = 0; ordinal != subordinateCount;
             ++ordinal) {
          ServiceRequestSignals request =
              requestInput(accessor, subordinateEndpoints[ordinal]->ports);
          const FirstServiceByteAddress firstAddress =
              firstServiceByteAddress(bodyBuilder, location, request,
                                      memoryServiceLayout, *arithmetic);
          // A single-target domain has no service-target bits; the decoded
          // field then reads as the one-bit zero `extract` yields for an
          // empty slice, and the selection starts from the same width.
          mlir::Value selectedTarget = zero(
              bodyBuilder, location, std::max(1U, layout.serviceTargetBitCount));
          mlir::Value selectedBase =
              zero(bodyBuilder, location, arithmetic->byteAddressWidthBits);
          mlir::Value selectedAny = bitConstant(bodyBuilder, location, false);
          const auto &declaration =
              connectivity->subordinateEndpoints()[ordinal];
          const auto &providerRows = layout.providerRows[ordinal];
          for (const auto &providerRow : providerRows) {
            // An address whose exact expression overflows the byte-address
            // domain selects no provider row, however its wrapped low bits
            // would compare.
            mlir::Value matches =
                andValues(bodyBuilder, location,
                          {memoryActive,
                           selectedBit(bodyBuilder, location, field,
                                       providerRow.bitOffset),
                           circt::comb::createOrFoldNot(bodyBuilder, location,
                                                        selectedAny),
                           firstAddress.exact});
            for (auto [matchOrdinal, matchField] :
                 llvm::enumerate(declaration.matchFields)) {
              const std::uint64_t offset =
                  providerRow.matchOffsets[matchOrdinal];
              mlir::Value fieldMatches =
                  bitConstant(bodyBuilder, location, false);
              switch (matchField) {
              case ::fabric::MemoryProviderMatchField::Range: {
                mlir::Value base =
                    extract(bodyBuilder, location, field, offset, 64);
                mlir::Value size =
                    extract(bodyBuilder, location, field, offset + 64, 64);
                mlir::Value end = circt::comb::AddOp::create(
                    bodyBuilder, location, base, size, true);
                fieldMatches = andValues(
                    bodyBuilder, location,
                    {circt::comb::ICmpOp::create(
                         bodyBuilder, location, circt::comb::ICmpPredicate::uge,
                         firstAddress.byteAddress, base, true),
                     circt::comb::ICmpOp::create(
                         bodyBuilder, location, circt::comb::ICmpPredicate::ult,
                         firstAddress.byteAddress, end, true)});
                break;
              }
              case ::fabric::MemoryProviderMatchField::Prefix: {
                mlir::Value value =
                    extract(bodyBuilder, location, field, offset, 64);
                mlir::Value length =
                    extract(bodyBuilder, location, field, offset + 64, 7);
                for (std::uint32_t prefix = 0; prefix <= 64; ++prefix) {
                  const std::uint64_t mask =
                      prefix == 0 ? 0
                      : prefix == 64
                          ? std::numeric_limits<std::uint64_t>::max()
                          : ~((std::uint64_t(1) << (64 - prefix)) - 1);
                  mlir::Value masked = circt::comb::AndOp::create(
                      bodyBuilder, location, firstAddress.byteAddress,
                      constant(bodyBuilder, location,
                               arithmetic->byteAddressWidthBits, mask),
                      true);
                  fieldMatches = circt::comb::OrOp::create(
                      bodyBuilder, location, fieldMatches,
                      andValues(bodyBuilder, location,
                                {equals(bodyBuilder, location, length, prefix),
                                 circt::comb::ICmpOp::create(
                                     bodyBuilder, location,
                                     circt::comb::ICmpPredicate::eq, masked,
                                     value, true)}));
                }
                break;
              }
              case ::fabric::MemoryProviderMatchField::Context:
              case ::fabric::MemoryProviderMatchField::AddressSpace:
                llvm_unreachable(
                    "unsupported provider match passed profile validation");
              }
              matches =
                  andValues(bodyBuilder, location, {matches, fieldMatches});
            }
            selectedTarget = circt::comb::MuxOp::create(
                bodyBuilder, location, matches,
                extract(bodyBuilder, location, field,
                        providerRow.serviceTargetOffset,
                        layout.serviceTargetBitCount),
                selectedTarget, true);
            if (declaration.addressTransform ==
                ::fabric::MemoryProviderAddressTransform::ConstantBaseOffset)
              selectedBase = circt::comb::MuxOp::create(
                  bodyBuilder, location, matches,
                  extract(bodyBuilder, location, field,
                          providerRow.baseOffsetOffset, 64),
                  selectedBase, true);
            selectedAny = circt::comb::OrOp::create(bodyBuilder, location,
                                                    selectedAny, matches);
          }
          request.baseAddress = circt::comb::AddOp::create(
              bodyBuilder, location, request.baseAddress, selectedBase, true);
          mlir::Value available = circt::comb::OrOp::create(
              bodyBuilder, location,
              circt::comb::createOrFoldNot(bodyBuilder, location,
                                           subordinateBusy[ordinal]),
              subordinateReleased[ordinal]);
          request.valid = andValues(bodyBuilder, location,
                                    {request.valid, selectedAny, available});
          requests.push_back(std::move(request));
          targetValid.push_back(selectedAny);
          targetCode.push_back(selectedTarget);
        }

        metrics.requestFormationOperations = blockOperationCount(bodyBuilder) -
                                             metrics.beginOperations -
                                             metrics.rowTransportOperations;
        std::vector<mlir::Value> localFired(
            requesterCount, bitConstant(bodyBuilder, location, false));
        ServiceRequestSignals localRequest =
            zeroRequest(bodyBuilder, location, memoryServiceLayout);
        std::optional<LocalRequestProjection> localProjection;
        if (hasLocalService && requesterCount != 0) {
          std::vector<mlir::Value> candidates;
          candidates.reserve(requesterCount);
          for (std::uint32_t requester = 0; requester != requesterCount;
               ++requester)
            candidates.push_back(andValues(
                bodyBuilder, location,
                {requests[requester].valid, targetValid[requester],
                 equals(bodyBuilder, location, targetCode[requester], 0)}));
          circt::Backedge cursorNext =
              backedges.get(bodyBuilder.getIntegerType(requesterCursorWidth));
          mlir::Value cursor = createRegister(
              bodyBuilder, location, cursorNext, accessor.getInput("clock"),
              accessor.getInput("reset"), llvm::APInt(requesterCursorWidth, 0),
              "local_service_cursor", clockReset.asynchronousReset);
          std::vector<mlir::Value> probed =
              roundRobinSelection(bodyBuilder, location, candidates, cursor);
          for (std::uint32_t requester = 0; requester != requesterCount;
               ++requester)
            localRequest = muxRequest(bodyBuilder, location, probed[requester],
                                      requests[requester], localRequest);
          localProjection = projectLocalRequest(
              bodyBuilder, location, localRequest, *serviceAccessCases,
              dataBytes, memoryServiceLayout, *arithmetic);
          for (std::uint32_t requester = 0; requester != requesterCount;
               ++requester)
            localFired[requester] =
                andValues(bodyBuilder, location,
                          {probed[requester], localProjection->legal});
          cursorNext.setValue(
              nextCursor(bodyBuilder, location, cursor, probed));
        }
        if (hasLocalService && !localProjection)
          localProjection = projectLocalRequest(
              bodyBuilder, location, localRequest, *serviceAccessCases,
              dataBytes, memoryServiceLayout, *arithmetic);

        std::vector<mlir::Value> managerFired(
            requesterCount, bitConstant(bodyBuilder, location, false));
        std::vector<mlir::Value> managerCompleted(
            requesterCount, bitConstant(bodyBuilder, location, false));
        std::vector<mlir::Value> managerResponseData(
            requesterCount,
            zero(bodyBuilder, location, memoryServiceLayout.dataWidthBits));
        for (std::uint32_t manager = 0; manager != managerEndpoints.size();
             ++manager) {
          const auto &ports = managerEndpoints[manager]->ports;
          if (requesterCount == 0) {
            setRequestOutputs(
                accessor, ports,
                zeroRequest(bodyBuilder, location, memoryServiceLayout));
            accessor.setOutput(ports.responseReady.getName(),
                               bitConstant(bodyBuilder, location, false));
            continue;
          }
          const std::uint64_t code =
              static_cast<std::uint64_t>(hasLocalService) + manager;
          std::vector<mlir::Value> candidates;
          candidates.reserve(requesterCount);
          for (std::uint32_t requester = 0; requester != requesterCount;
               ++requester)
            candidates.push_back(andValues(
                bodyBuilder, location,
                {requests[requester].valid, targetValid[requester],
                 equals(bodyBuilder, location, targetCode[requester], code)}));

          circt::Backedge ownedNext = backedges.get(bodyBuilder.getI1Type());
          circt::Backedge acceptedNext = backedges.get(bodyBuilder.getI1Type());
          circt::Backedge ownerNext =
              backedges.get(bodyBuilder.getIntegerType(requesterCursorWidth));
          mlir::Value owned = createRegister(
              bodyBuilder, location, ownedNext, accessor.getInput("clock"),
              accessor.getInput("reset"), llvm::APInt(1, 0),
              "manager_owned_" + std::to_string(manager),
              clockReset.asynchronousReset);
          mlir::Value accepted = createRegister(
              bodyBuilder, location, acceptedNext, accessor.getInput("clock"),
              accessor.getInput("reset"), llvm::APInt(1, 0),
              "manager_accepted_" + std::to_string(manager),
              clockReset.asynchronousReset);
          mlir::Value owner = createRegister(
              bodyBuilder, location, ownerNext, accessor.getInput("clock"),
              accessor.getInput("reset"), llvm::APInt(requesterCursorWidth, 0),
              "manager_owner_" + std::to_string(manager),
              clockReset.asynchronousReset);
          circt::Backedge cursorNext =
              backedges.get(bodyBuilder.getIntegerType(requesterCursorWidth));
          mlir::Value cursor = createRegister(
              bodyBuilder, location, cursorNext, accessor.getInput("clock"),
              accessor.getInput("reset"), llvm::APInt(requesterCursorWidth, 0),
              "manager_cursor_" + std::to_string(manager),
              clockReset.asynchronousReset);
          std::vector<mlir::Value> selectedNew =
              roundRobinSelection(bodyBuilder, location, candidates, cursor);
          std::vector<mlir::Value> selected(requesterCount);
          mlir::Value anySelected = bitConstant(bodyBuilder, location, false);
          ServiceRequestSignals outgoing =
              zeroRequest(bodyBuilder, location, memoryServiceLayout);
          for (std::uint32_t requester = 0; requester != requesterCount;
               ++requester) {
            selected[requester] = circt::comb::MuxOp::create(
                bodyBuilder, location, owned,
                equals(bodyBuilder, location, owner, requester),
                selectedNew[requester], true);
            anySelected = circt::comb::OrOp::create(
                bodyBuilder, location, anySelected, selected[requester]);
            outgoing = muxRequest(bodyBuilder, location, selected[requester],
                                  requests[requester], outgoing);
          }
          outgoing.valid = andValues(
              bodyBuilder, location,
              {outgoing.valid,
               circt::comb::createOrFoldNot(bodyBuilder, location, accepted)});
          setRequestOutputs(accessor, ports, outgoing);
          mlir::Value requestFire =
              andValues(bodyBuilder, location,
                        {outgoing.valid,
                         accessor.getInput(ports.requestReady.getName())});
          std::vector<mlir::Value> firedThisManager(requesterCount);
          for (std::uint32_t requester = 0; requester != requesterCount;
               ++requester) {
            firedThisManager[requester] = andValues(
                bodyBuilder, location, {selected[requester], requestFire});
            managerFired[requester] = circt::comb::OrOp::create(
                bodyBuilder, location, managerFired[requester],
                firedThisManager[requester]);
          }
          mlir::Value responseReady =
              andValues(bodyBuilder, location, {owned, accepted});
          accessor.setOutput(ports.responseReady.getName(), responseReady);
          mlir::Value responseFire =
              andValues(bodyBuilder, location,
                        {responseReady,
                         accessor.getInput(ports.responseValid.getName())});
          for (std::uint32_t requester = 0; requester != requesterCount;
               ++requester) {
            mlir::Value returns =
                andValues(bodyBuilder, location,
                          {responseFire,
                           equals(bodyBuilder, location, owner, requester)});
            managerCompleted[requester] = circt::comb::OrOp::create(
                bodyBuilder, location, managerCompleted[requester], returns);
            managerResponseData[requester] = circt::comb::MuxOp::create(
                bodyBuilder, location, returns,
                accessor.getInput(ports.responseData.getName()),
                managerResponseData[requester], true);
          }
          mlir::Value acquire = andValues(
              bodyBuilder, location,
              {circt::comb::createOrFoldNot(bodyBuilder, location, owned),
               anySelected});
          mlir::Value selectedOwner = owner;
          for (std::uint32_t requester = 0; requester != requesterCount;
               ++requester)
            selectedOwner = circt::comb::MuxOp::create(
                bodyBuilder, location, selectedNew[requester],
                constant(bodyBuilder, location, requesterCursorWidth,
                         requester),
                selectedOwner, true);
          ownerNext.setValue(circt::comb::MuxOp::create(
              bodyBuilder, location, acquire, selectedOwner, owner, true));
          ownedNext.setValue(
              andValues(bodyBuilder, location,
                        {orValues(bodyBuilder, location, {owned, anySelected}),
                         circt::comb::createOrFoldNot(bodyBuilder, location,
                                                      responseFire)}));
          acceptedNext.setValue(andValues(
              bodyBuilder, location,
              {orValues(bodyBuilder, location, {accepted, requestFire}),
               circt::comb::createOrFoldNot(bodyBuilder, location,
                                            responseFire)}));
          cursorNext.setValue(
              nextCursor(bodyBuilder, location, cursor, firedThisManager));
        }

        metrics.arbitrationOperations =
            blockOperationCount(bodyBuilder) - metrics.beginOperations -
            metrics.rowTransportOperations - metrics.requestFormationOperations;
        std::vector<mlir::Value> issued(
            requesterCount, bitConstant(bodyBuilder, location, false));
        std::vector<mlir::Value> completedRequest(
            requesterCount, bitConstant(bodyBuilder, location, false));
        for (std::uint32_t requester = 0; requester != requesterCount;
             ++requester) {
          issued[requester] =
              orValues(bodyBuilder, location,
                       {localFired[requester], managerFired[requester]});
          completedRequest[requester] =
              orValues(bodyBuilder, location,
                       {localFired[requester], managerCompleted[requester]});
        }

        for (std::uint32_t row = 0; row != rowCount; ++row)
          for (std::uint32_t role = 0; role != layout.roleCount; ++role) {
            MemoryOperandQueueRuntime &queue = operandQueues[row][role];
            mlir::Value queuedSource = rows[row].sourceInternal[role];
            if (layout.schedule == ::fabric::Schedule::Temporal)
              queuedSource = rows[row].sourcePresent[role];
            mlir::Value dequeue = andValues(
                bodyBuilder, location,
                {issued[row], rows[row].sourcePresent[role], queuedSource});
            queue.dequeue.setValue(dequeue);
            queue.occupiedNext.setValue(
                orValues(bodyBuilder, location,
                         {andValues(bodyBuilder, location,
                                    {queue.occupied,
                                     circt::comb::createOrFoldNot(
                                         bodyBuilder, location, dequeue)}),
                          queue.enqueue}));
            queue.dataNext.setValue(circt::comb::MuxOp::create(
                bodyBuilder, location, queue.enqueue, queue.enqueueData,
                queue.data, true));
          }

        for (const EndpointPlan &endpoint : *endpoints) {
          if (endpoint.direction != fabric::FabricPortDirection::Input)
            continue;
          if (layout.schedule == ::fabric::Schedule::Temporal) {
            const auto terms = inputReadyTerms.find(endpoint.endpoint.ordinal);
            accessor.setOutput(
                endpoint.ready.getName(),
                terms == inputReadyTerms.end()
                    ? bitConstant(bodyBuilder, location, false)
                    : orValues(bodyBuilder, location, terms->second));
            continue;
          }
          mlir::Value ready = bitConstant(bodyBuilder, location, false);
          for (std::uint32_t row = 0; row != rowCount; ++row)
            for (std::uint32_t role = 0; role != layout.roleCount; ++role) {
              mlir::Value selected = andValues(
                  bodyBuilder, location,
                  {issued[row], rows[row].sourcePresent[role],
                   circt::comb::createOrFoldNot(bodyBuilder, location,
                                                rows[row].sourceInternal[role]),
                   equals(bodyBuilder, location, rows[row].sourceEndpoint[role],
                          endpoint.endpoint.ordinal)});
              ready = circt::comb::OrOp::create(bodyBuilder, location, ready,
                                                selected);
            }
          accessor.setOutput(endpoint.ready.getName(), ready);
        }

        metrics.issueReadinessOperations =
            blockOperationCount(bodyBuilder) - metrics.beginOperations -
            metrics.rowTransportOperations -
            metrics.requestFormationOperations - metrics.arbitrationOperations;
        mlir::Value assembled = zero(bodyBuilder, location, dataWidth);
        if (hasLocalService) {
          auto memoryType = circt::seq::FirMemType::get(
              bodyBuilder.getContext(), capacity, 8, std::nullopt);
          auto storage = circt::seq::FirMemOp::create(
              bodyBuilder, location, memoryType, 0, 1, circt::seq::RUW::Old,
              circt::seq::WUW::PortOrder,
              bodyBuilder.getStringAttr("local_storage"),
              circt::hw::InnerSymAttr{}, circt::seq::FirMemInitAttr{},
              mlir::StringAttr{}, mlir::Attribute{});
          const unsigned storageAddressWidth = indexWidth(capacity);
          std::vector<mlir::Value> readBytes;
          readBytes.reserve(dataBytes);
          for (std::uint64_t byte = 0; byte != dataBytes; ++byte) {
            const AddressedByte &selected = localProjection->bytes[byte];
            mlir::Value localAddress = circt::comb::ExtractOp::create(
                bodyBuilder, location, selected.address, 0,
                storageAddressWidth);
            mlir::Value readEnable =
                andValues(bodyBuilder, location,
                          {selected.active, equals(bodyBuilder, location,
                                                   localRequest.kind, 0)});
            mlir::Value readByte = circt::seq::FirMemReadOp::create(
                bodyBuilder, location, storage, localAddress,
                accessor.getInput("clock"), readEnable);
            readBytes.push_back(circt::comb::MuxOp::create(
                bodyBuilder, location, readEnable, readByte,
                zero(bodyBuilder, location, 8), true));
            mlir::Value writeEnable =
                andValues(bodyBuilder, location,
                          {selected.active, equals(bodyBuilder, location,
                                                   localRequest.kind, 1)});
            mlir::Value writeByte =
                extract(bodyBuilder, location, localRequest.data, byte * 8,
                        std::min<std::uint64_t>(8, dataWidth - byte * 8));
            writeByte = adaptWidth(bodyBuilder, location, writeByte, 8);
            circt::seq::FirMemWriteOp::create(
                bodyBuilder, location, storage, localAddress,
                accessor.getInput("clock"), writeEnable, writeByte,
                mlir::Value{});
          }
          llvm::SmallVector<mlir::Value> highToLow;
          for (mlir::Value byte : llvm::reverse(readBytes))
            highToLow.push_back(byte);
          assembled = highToLow.size() == 1
                          ? highToLow.front()
                          : mlir::Value(circt::comb::ConcatOp::create(
                                bodyBuilder, location, highToLow));
          assembled = adaptWidth(bodyBuilder, location, assembled, dataWidth);
        }

        metrics.localStorageOperations =
            blockOperationCount(bodyBuilder) - metrics.beginOperations -
            metrics.rowTransportOperations -
            metrics.requestFormationOperations - metrics.arbitrationOperations -
            metrics.issueReadinessOperations;
        for (std::uint32_t row = 0; row != rowCount; ++row) {
          occupiedNext[row].setValue(
              orValues(bodyBuilder, location,
                       {andValues(bodyBuilder, location,
                                  {occupied[row],
                                   circt::comb::createOrFoldNot(
                                       bodyBuilder, location, released[row])}),
                        issued[row]}));
          completedNext[row].setValue(
              orValues(bodyBuilder, location,
                       {andValues(bodyBuilder, location,
                                  {completed[row],
                                   circt::comb::createOrFoldNot(
                                       bodyBuilder, location, released[row])}),
                        completedRequest[row]}));
          mlir::Value completionData = circt::comb::MuxOp::create(
              bodyBuilder, location, localFired[row], assembled,
              managerResponseData[row], true);
          resultDataNext[row].setValue(circt::comb::MuxOp::create(
              bodyBuilder, location, completedRequest[row], completionData,
              resultData[row], true));
        }
        for (std::uint32_t ordinal = 0; ordinal != subordinateCount;
             ++ordinal) {
          const std::uint32_t requester = rowCount + ordinal;
          accessor.setOutput(
              subordinateEndpoints[ordinal]->ports.requestReady.getName(),
              issued[requester]);
          subordinateBusyNext[ordinal].setValue(orValues(
              bodyBuilder, location,
              {andValues(
                   bodyBuilder, location,
                   {subordinateBusy[ordinal],
                    circt::comb::createOrFoldNot(
                        bodyBuilder, location, subordinateReleased[ordinal])}),
               issued[requester]}));
          subordinateCompletedNext[ordinal].setValue(orValues(
              bodyBuilder, location,
              {andValues(
                   bodyBuilder, location,
                   {subordinateCompleted[ordinal],
                    circt::comb::createOrFoldNot(
                        bodyBuilder, location, subordinateReleased[ordinal])}),
               completedRequest[requester]}));
          mlir::Value completionData = circt::comb::MuxOp::create(
              bodyBuilder, location, localFired[requester], assembled,
              managerResponseData[requester], true);
          subordinateDataNext[ordinal].setValue(circt::comb::MuxOp::create(
              bodyBuilder, location, completedRequest[requester],
              completionData, subordinateData[ordinal], true));
        }
        metrics.completionOperations =
            blockOperationCount(bodyBuilder) - metrics.beginOperations -
            metrics.rowTransportOperations -
            metrics.requestFormationOperations - metrics.arbitrationOperations -
            metrics.issueReadinessOperations - metrics.localStorageOperations;
      });
  if (materializationError)
    return invalid(*materializationError);
  if (invocationDiagnosticEnabled(DiagnosticVerbosity::Summary)) {
    const std::uint64_t scanLevels =
        obligationCount < 2 ? 0 : llvm::Log2_64_Ceil(obligationCount);
    const std::uint64_t actualOperations =
        metrics.rowTransportOperations + metrics.requestFormationOperations +
        metrics.arbitrationOperations + metrics.issueReadinessOperations +
        metrics.localStorageOperations + metrics.completionOperations;
    emitInvocationDiagnostic(
        DiagnosticVerbosity::Summary,
        InvocationDiagnosticStage::HardwareConfiguration,
        InvocationDiagnosticEvent::Statistics, [&] {
          return llvm::json::Value(llvm::json::Object{
              {"statistics_kind", "rtl_memory_materialization_shape"},
              {"materialization_key", materializationKey.str()},
              {"memory_ordinal", memory.id()},
              {"operation_row_count", rowCount},
              {"operation_access_case_count", accessCases->size()},
              {"service_access_case_count", serviceAccessCases->size()},
              {"role_count", roleCount},
              {"internal_connection_count", connectionCount},
              {"input_endpoint_count", inputEndpointCount},
              {"output_endpoint_count", outputEndpointCount},
              {"manager_endpoint_count", managerEndpoints.size()},
              {"subordinate_endpoint_count", subordinateEndpoints.size()},
              {"data_bytes", dataBytes},
              {"predicted_row_access_predicates",
               saturatedProduct({rowCount, accessCases->size()})},
              {"predicted_role_input_predicates",
               saturatedProduct({rowCount, roleCount, inputEndpointCount})},
              {"predicted_internal_queue_predicates",
               saturatedProduct({connectionCount, rowCount, roleCount})},
              {"semantic_result_obligation_count", semanticObligationCount},
              {"predicted_result_obligation_count", obligationCount},
              {"predicted_atomic_scan_predicates",
               saturatedProduct({2, obligationCount, scanLevels})},
              {"predicted_local_legality_byte_cases",
               saturatedProduct({serviceAccessCases->size(), dataBytes})},
              {"actual_row_transport_operations",
               metrics.rowTransportOperations},
              {"actual_request_formation_operations",
               metrics.requestFormationOperations},
              {"actual_arbitration_operations", metrics.arbitrationOperations},
              {"actual_issue_readiness_operations",
               metrics.issueReadinessOperations},
              {"actual_local_storage_operations",
               metrics.localStorageOperations},
              {"actual_completion_operations", metrics.completionOperations},
              {"actual_module_body_operations", actualOperations}});
        });
  }
  std::string implementationKey;
  llvm::raw_string_ostream keyStream(implementationKey);
  module.print(keyStream);
  keyStream.flush();
  replaceAll(implementationKey, memoryName(memory), "MEMORY_MODULE");
  return MemoryModule{memory,
                      module,
                      std::move(*endpoints),
                      std::move(*memoryEndpoints),
                      std::move(implementationKey),
                      std::move(*configuration),
                      std::move(*decoder)};
}

} // namespace

llvm::Expected<std::vector<MemoryModule>>
buildMemoryModules(mlir::OpBuilder &builder, mlir::Location location,
                   fabric::SpatialCoreOccurrenceRef spatialCore,
                   const fabric::FabricArtifactView &fabric,
                   const ConfigurationABI &configurationAbi,
                   const ConfigurationTransportLayout &transportLayout,
                   const ClockResetPlan &clockReset,
                   const PortableMemoryServiceLayout &memoryServiceLayout,
                   llvm::StringRef materializationKey) {
  std::vector<MemoryModule> result;
  result.reserve(fabric.memoryOccurrences().size());
  std::map<std::string, circt::hw::HWModuleOp> definitions;
  for (fabric::FabricMemoryOccurrenceRef memory : fabric.memoryOccurrences()) {
    auto module =
        buildMemoryModule(builder, location, spatialCore, fabric,
                          configurationAbi, transportLayout, clockReset,
                          memoryServiceLayout, memory, materializationKey);
    if (!module)
      return module.takeError();
    if (llvm::Error error = verifyConfigurationValuePort(
            module->module, module->configurationDecoder))
      return std::move(error);
    auto definition = definitions.find(module->implementationKey);
    if (definition == definitions.end()) {
      definitions.emplace(module->implementationKey, module->module);
    } else {
      module->module.erase();
      module->module = definition->second;
      if (llvm::Error error = verifyConfigurationValuePort(
              module->module, module->configurationDecoder))
        return std::move(error);
    }
    module->implementationKey.clear();
    result.push_back(std::move(*module));
  }
  return result;
}

} // namespace loom::hardware::rtl::hierarchy
