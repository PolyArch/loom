#include "ADG/MemoryLibrary.h"

#include "CatalogCapabilities.h"

#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/IR/MemoryActorContractDomain.h"
#include "Fabric/IR/MemoryCapabilityDomains.h"
#include "Fabric/IR/MemoryConnectivityContract.h"
#include "Fabric/IR/MemoryOperationPort.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/IR/ResourceContract.h"
#include "Fabric/IR/SystemServiceContract.h"

#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom::adg {
namespace {

using ::dataflow::OperationSchemaId;
using ::dataflow::semantics::MemoryAccessForm;
using ::dataflow::semantics::MemoryMaskForm;
using ::dataflow::semantics::ServiceValueRole;

enum class CatalogMemoryDomain { Hybrid32, General64 };
enum class AccessProjectionDomain { Direct, Indexed, All };

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "adg_memory_library_invalid: " + message);
}

llvm::Expected<::fabric::UnsignedDomain> singleton(std::uint64_t value) {
  return ::fabric::UnsignedDomain::fromCanonical({{value, value}});
}

llvm::Expected<::fabric::UnsignedDomain>
scalarElementWidths(CatalogMemoryDomain domain) {
  if (domain == CatalogMemoryDomain::General64)
    return ::fabric::UnsignedDomain::fromCanonical(
        {{8, 8}, {16, 16}, {32, 32}, {64, 64}});
  return ::fabric::UnsignedDomain::fromCanonical({{8, 8}, {16, 16}, {32, 32}});
}

llvm::ArrayRef<std::uint32_t> catalogElementWidths(CatalogMemoryDomain domain) {
  static constexpr std::array<std::uint32_t, 3> hybrid = {8, 16, 32};
  static constexpr std::array<std::uint32_t, 4> general = {8, 16, 32, 64};
  return domain == CatalogMemoryDomain::General64
             ? llvm::ArrayRef<std::uint32_t>(general)
             : llvm::ArrayRef<std::uint32_t>(hybrid);
}

llvm::Expected<::fabric::AlignmentDomain> allAlignments() {
  auto exponents = ::fabric::UnsignedDomain::fromCanonical({{0, 63}});
  if (!exponents)
    return exponents.takeError();
  return ::fabric::AlignmentDomain::create(std::move(*exponents));
}

template <typename Enum>
llvm::Expected<::fabric::ClosedEnumDomain<Enum>>
enumDomain(std::initializer_list<Enum> values) {
  return ::fabric::ClosedEnumDomain<Enum>::fromCanonical(values);
}

llvm::Expected<::fabric::MemoryAccessClass>
elementAccess(bool reads, CatalogMemoryDomain domain,
              dataflow::semantics::MemoryAddressForm addressForm) {
  auto widths = scalarElementWidths(domain);
  if (!widths)
    return widths.takeError();
  auto lanes = singleton(1);
  if (!lanes)
    return lanes.takeError();
  auto alignments = allAlignments();
  if (!alignments)
    return alignments.takeError();
  auto read = enumDomain<::fabric::ReadSubwordSemantics>(
      {reads ? ::fabric::ReadSubwordSemantics::ZeroExtend
             : ::fabric::ReadSubwordSemantics::NotApplicable});
  if (!read)
    return read.takeError();
  auto write = enumDomain<::fabric::WriteSubwordSemantics>(
      {reads ? ::fabric::WriteSubwordSemantics::NotApplicable
             : ::fabric::WriteSubwordSemantics::ByteEnable});
  if (!write)
    return write.takeError();
  return ::fabric::MemoryAccessClass::create(
      MemoryAccessForm::Element, std::move(*widths), std::move(*lanes),
      {{MemoryMaskForm::Absent,
        ::fabric::InactiveLaneSemantics::NotApplicable}},
      std::move(*alignments), std::move(*read), std::move(*write), addressForm,
      addressForm == dataflow::semantics::MemoryAddressForm::PointerAddressed
          ? ::loom::adg::detail::catalogPointerFormats()
          : ::fabric::PointerFormatRelation{},
      ::loom::adg::detail::catalogPointerFormats());
}

llvm::Expected<::fabric::MemoryAccessClass>
vectorAccess(bool reads, MemoryAccessForm accessForm,
             dataflow::semantics::MemoryAddressForm addressForm,
             std::uint32_t elementWidth, std::uint64_t maximumLanes,
             llvm::ArrayRef<::fabric::MaskInactivePair> masks) {
  auto widths = singleton(elementWidth);
  if (!widths)
    return widths.takeError();
  auto lanes = ::fabric::UnsignedDomain::fromCanonical({{2, maximumLanes}});
  if (!lanes)
    return lanes.takeError();
  auto alignments = allAlignments();
  if (!alignments)
    return alignments.takeError();
  auto read = enumDomain<::fabric::ReadSubwordSemantics>(
      {reads ? ::fabric::ReadSubwordSemantics::ZeroExtend
             : ::fabric::ReadSubwordSemantics::NotApplicable});
  if (!read)
    return read.takeError();
  auto write = enumDomain<::fabric::WriteSubwordSemantics>(
      {reads ? ::fabric::WriteSubwordSemantics::NotApplicable
             : ::fabric::WriteSubwordSemantics::ByteEnable});
  if (!write)
    return write.takeError();
  return ::fabric::MemoryAccessClass::create(
      accessForm, std::move(*widths), std::move(*lanes), masks,
      std::move(*alignments), std::move(*read), std::move(*write), addressForm,
      addressForm == dataflow::semantics::MemoryAddressForm::PointerAddressed
          ? ::loom::adg::detail::catalogPointerFormats()
          : ::fabric::PointerFormatRelation{},
      ::loom::adg::detail::catalogPointerFormats());
}

llvm::Expected<::fabric::ParameterizedMemoryAccessDomain>
accessDomain(bool reads, CatalogMemoryDomain domain,
             const MemoryAccessDomainParameters &parameters,
             AccessProjectionDomain projection) {
  using dataflow::semantics::MemoryAddressForm;
  if (parameters.dataPayloadBits == 0 || parameters.maskPayloadBits == 0)
    return invalid("memory access-domain widths must be positive");
  if (projection != AccessProjectionDomain::Direct &&
      !parameters.indexedAddressPayloadBits)
    return invalid("indexed memory domain requires an indexed address width");

  std::vector<::fabric::MemoryAccessClass> classes;
  constexpr std::array addressForms = {MemoryAddressForm::RootRelative,
                                       MemoryAddressForm::PointerAddressed};
  if (projection != AccessProjectionDomain::Indexed) {
    for (MemoryAddressForm addressForm : addressForms) {
      auto element = elementAccess(reads, domain, addressForm);
      if (!element)
        return element.takeError();
      classes.push_back(std::move(*element));
    }
  }

  auto appendVectorClasses = [&](MemoryAccessForm accessForm,
                                 MemoryAddressForm addressForm,
                                 std::uint32_t elementWidth,
                                 std::uint64_t maximumLanes) -> llvm::Error {
    if (maximumLanes < 2)
      return llvm::Error::success();
    const ::fabric::MaskInactivePair absent{
        MemoryMaskForm::Absent, ::fabric::InactiveLaneSemantics::NotApplicable};
    auto unmasked = vectorAccess(reads, accessForm, addressForm, elementWidth,
                                 maximumLanes, {absent});
    if (!unmasked)
      return unmasked.takeError();
    classes.push_back(std::move(*unmasked));

    const std::uint64_t maskedMaximum =
        std::min<std::uint64_t>(maximumLanes, parameters.maskPayloadBits);
    if (maskedMaximum < 2)
      return llvm::Error::success();
    const ::fabric::MaskInactivePair dynamic{
        MemoryMaskForm::Dynamic,
        reads ? ::fabric::InactiveLaneSemantics::SuppressAndZeroFill
              : ::fabric::InactiveLaneSemantics::Suppress};
    auto masked = vectorAccess(reads, accessForm, addressForm, elementWidth,
                               maskedMaximum, {dynamic});
    if (!masked)
      return masked.takeError();
    classes.push_back(std::move(*masked));
    return llvm::Error::success();
  };

  for (std::uint32_t width : catalogElementWidths(domain)) {
    const std::uint64_t dataLanes = parameters.dataPayloadBits / width;
    if (projection != AccessProjectionDomain::Indexed)
      for (MemoryAddressForm addressForm : addressForms)
        if (llvm::Error error = appendVectorClasses(
                MemoryAccessForm::Contiguous, addressForm, width, dataLanes))
          return std::move(error);
    if (projection != AccessProjectionDomain::Direct) {
      const std::uint64_t addressLanes =
          *parameters.indexedAddressPayloadBits / 32;
      const std::uint64_t indexedLanes = std::min(dataLanes, addressLanes);
      for (MemoryAddressForm addressForm : addressForms)
        if (llvm::Error error = appendVectorClasses(
                MemoryAccessForm::Indexed, addressForm, width, indexedLanes))
          return std::move(error);
    }
  }
  if (classes.empty())
    return invalid("memory recipe admits no access class");
  return ::fabric::ParameterizedMemoryAccessDomain::create(classes);
}

llvm::Expected<::fabric::MemoryActorContractDomain> actorDomain(bool reads) {
  ::fabric::MemoryActorContractClause plain =
      ::fabric::LoadStorePlainContractClause{{false}};
  return ::fabric::MemoryActorContractDomain::create(
      reads ? OperationSchemaId::DataflowLoad
            : OperationSchemaId::DataflowStore,
      {plain});
}

std::vector<::fabric::InternalTransactionDeclaration>
transactions(std::uint32_t count) {
  return std::vector<::fabric::InternalTransactionDeclaration>(
      count, ::fabric::InternalTransactionDeclaration{{::fabric::ClaimKey(0)}});
}

llvm::Expected<::fabric::ResourceContract>
operationPortResourceContract(std::optional<std::uint32_t> indexedLaneCount) {
  ::fabric::ResourceContractDeclaration declaration;
  declaration.states = {
      {::fabric::StateKey(0),
       {{::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(1),
         ::fabric::CapacityUnits(0)}}}};
  const std::uint32_t patternCount = indexedLaneCount ? 2 : 1;
  for (std::uint32_t ordinal = 0; ordinal != patternCount; ++ordinal)
    declaration.requesters.push_back(::fabric::RequesterKey(ordinal));
  declaration.eligibilityCount = patternCount;
  declaration.eventCount = 2;
  declaration.timingContracts = {{::fabric::TimingContractKey(0), {0, 1}}};
  for (std::uint32_t ordinal = 0; ordinal != patternCount; ++ordinal)
    declaration.usePatterns.push_back(
        {::fabric::UsePatternKey(ordinal),
         ::fabric::RequesterKey(ordinal),
         ::fabric::EligibilityKey(ordinal),
         ::fabric::EventKey(0),
         ::fabric::EventKey(1),
         std::nullopt,
         ::fabric::TimingContractKey(0),
         {{::fabric::ClaimKey(0), ::fabric::StateKey(0),
           ::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(1)}},
         transactions(ordinal == 0 ? 1 : *indexedLaneCount)});
  if (indexedLaneCount)
    declaration.grantPolicy = ::fabric::RoundRobinDeclaration{
        {::fabric::RequesterKey(0), ::fabric::RequesterKey(1)},
        ::fabric::RequesterKey(0)};
  return ::fabric::ResourceContract::create(declaration);
}

llvm::Expected<::fabric::ResourceContract>
memoryServiceResourceContract(std::uint32_t maximumBeatCount) {
  if (maximumBeatCount == 0)
    return invalid("memory service requires a positive beat count");
  ::fabric::ResourceContractDeclaration declaration;
  declaration.states = {
      {::fabric::StateKey(0),
       {{::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(1),
         ::fabric::CapacityUnits(0)}}}};
  declaration.requesters = {::fabric::RequesterKey(0),
                            ::fabric::RequesterKey(1)};
  declaration.eligibilityCount = 2;
  declaration.eventCount = 2;
  declaration.timingContracts = {{::fabric::TimingContractKey(0), {0, 1}}};
  for (std::uint32_t ordinal = 0; ordinal != 2; ++ordinal)
    declaration.usePatterns.push_back(
        {::fabric::UsePatternKey(ordinal),
         ::fabric::RequesterKey(ordinal),
         ::fabric::EligibilityKey(ordinal),
         ::fabric::EventKey(0),
         ::fabric::EventKey(1),
         std::nullopt,
         ::fabric::TimingContractKey(0),
         {{::fabric::ClaimKey(0), ::fabric::StateKey(0),
           ::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(1)}},
         transactions(maximumBeatCount)});
  declaration.grantPolicy = ::fabric::RoundRobinDeclaration{
      {::fabric::RequesterKey(0), ::fabric::RequesterKey(1)},
      ::fabric::RequesterKey(0)};
  return ::fabric::ResourceContract::create(declaration);
}

struct MemoryEndpointLayout final {
  std::vector<std::uint32_t> inputWidths;
  std::vector<std::uint32_t> outputWidths;
  std::uint64_t readScalarAddress = 0;
  std::optional<std::uint64_t> readIndexedAddress;
  std::uint64_t readMask = 0;
  std::uint64_t readControl = 0;
  std::uint64_t writeScalarAddress = 0;
  std::optional<std::uint64_t> writeIndexedAddress;
  std::uint64_t writeData = 0;
  std::uint64_t writeMask = 0;
  std::uint64_t writeControl = 0;
  std::uint64_t readData = 0;
  std::uint64_t readCompletion = 0;
  std::uint64_t writeCompletion = 0;
};

MemoryEndpointLayout
endpointLayout(const MemoryInterfaceParameters &interface) {
  MemoryEndpointLayout layout;
  auto addInput = [&](std::uint32_t width) {
    const std::uint64_t ordinal = layout.inputWidths.size();
    layout.inputWidths.push_back(width);
    return ordinal;
  };
  layout.readScalarAddress = addInput(interface.scalarAddressPayloadBits);
  if (interface.accessDomain.indexedAddressPayloadBits)
    layout.readIndexedAddress =
        addInput(*interface.accessDomain.indexedAddressPayloadBits);
  layout.readMask = addInput(interface.accessDomain.maskPayloadBits);
  layout.readControl = addInput(0);
  layout.writeScalarAddress = addInput(interface.scalarAddressPayloadBits);
  if (interface.accessDomain.indexedAddressPayloadBits)
    layout.writeIndexedAddress =
        addInput(*interface.accessDomain.indexedAddressPayloadBits);
  layout.writeData = addInput(interface.accessDomain.dataPayloadBits);
  layout.writeMask = addInput(interface.accessDomain.maskPayloadBits);
  layout.writeControl = addInput(0);

  const std::uint64_t outputBase = layout.inputWidths.size();
  layout.readData = outputBase;
  layout.readCompletion = outputBase + 1;
  layout.writeCompletion = outputBase + 2;
  layout.outputWidths = {interface.accessDomain.dataPayloadBits, 0, 0};
  return layout;
}

std::vector<::fabric::MemoryTransportEndpointDescriptor>
endpointInventory(const MemoryEndpointLayout &layout,
                  std::optional<std::uint32_t> tagWidth) {
  using loom::fabric::FabricPortDirection;
  std::vector<::fabric::MemoryTransportEndpointDescriptor> endpoints;
  for (std::uint32_t width : layout.inputWidths)
    endpoints.push_back({FabricPortDirection::Input, width, tagWidth});
  for (std::uint32_t width : layout.outputWidths)
    endpoints.push_back({FabricPortDirection::Output, width, tagWidth});
  return endpoints;
}

std::uint32_t
maximumIndexedLaneCount(const MemoryAccessDomainParameters &parameters) {
  if (!parameters.indexedAddressPayloadBits)
    return 0;
  return std::min(parameters.dataPayloadBits / 8,
                  *parameters.indexedAddressPayloadBits / 32);
}

std::vector<::fabric::MemoryRoleEndpointBindingRecord>
roleBindings(bool reads, std::uint64_t address,
             const MemoryEndpointLayout &layout) {
  if (reads)
    return {{ServiceValueRole::Address, address},
            {ServiceValueRole::Data, layout.readData},
            {ServiceValueRole::Mask, layout.readMask},
            {ServiceValueRole::Control, layout.readControl},
            {ServiceValueRole::Completion, layout.readCompletion}};
  return {{ServiceValueRole::Address, address},
          {ServiceValueRole::Data, layout.writeData},
          {ServiceValueRole::Mask, layout.writeMask},
          {ServiceValueRole::Control, layout.writeControl},
          {ServiceValueRole::Completion, layout.writeCompletion}};
}

llvm::Expected<::fabric::MemoryOperationPortDeclaration> operationPort(
    mlir::MLIRContext &context, ::fabric::Schedule schedule,
    llvm::ArrayRef<::fabric::MemoryTransportEndpointDescriptor> endpoints,
    const MemoryEndpointLayout &layout, bool reads, CatalogMemoryDomain domain,
    const MemoryAccessDomainParameters &parameters) {
  const std::uint32_t indexedLanes = maximumIndexedLaneCount(parameters);
  if (parameters.indexedAddressPayloadBits && indexedLanes < 2)
    return invalid("indexed memory endpoint cannot carry two lane addresses");
  auto resources = operationPortResourceContract(
      parameters.indexedAddressPayloadBits
          ? std::optional<std::uint32_t>(indexedLanes)
          : std::nullopt);
  if (!resources)
    return resources.takeError();
  auto actors = actorDomain(reads);
  if (!actors)
    return actors.takeError();
  auto directAccesses =
      accessDomain(reads, domain, parameters, AccessProjectionDomain::Direct);
  if (!directAccesses)
    return directAccesses.takeError();

  const std::uint64_t scalarAddress =
      reads ? layout.readScalarAddress : layout.writeScalarAddress;
  std::vector<::fabric::MemoryCapabilityAlternativeRecord> alternatives;
  alternatives.push_back({*actors,
                          roleBindings(reads, scalarAddress, layout),
                          std::move(*directAccesses),
                          {::fabric::UsePatternKey(0)}});
  if (parameters.indexedAddressPayloadBits) {
    auto indexedAccesses = accessDomain(reads, domain, parameters,
                                        AccessProjectionDomain::Indexed);
    if (!indexedAccesses)
      return indexedAccesses.takeError();
    const std::uint64_t indexedAddress =
        reads ? *layout.readIndexedAddress : *layout.writeIndexedAddress;
    alternatives.push_back({std::move(*actors),
                            roleBindings(reads, indexedAddress, layout),
                            std::move(*indexedAccesses),
                            {::fabric::UsePatternKey(1)}});
  }

  std::vector<std::uint64_t> inventory;
  for (const auto &alternative : alternatives)
    for (const auto &binding : alternative.roleToEndpoint)
      inventory.push_back(binding.endpointOrdinal);
  llvm::sort(inventory);
  inventory.erase(std::unique(inventory.begin(), inventory.end()),
                  inventory.end());
  std::vector<::fabric::MemoryOperationPatternRecord> patterns = {
      {::fabric::MemoryPortTransactionProjection::Direct}};
  if (parameters.indexedAddressPayloadBits)
    patterns.push_back(
        {::fabric::MemoryPortTransactionProjection::ActiveLanesRowMajor});
  ::fabric::MemoryOperationPortDeclaration declaration{
      std::move(inventory), std::move(*resources), std::move(patterns),
      std::move(alternatives)};
  auto record = ::fabric::MemoryOperationPortRecord::create(
      &context, schedule, endpoints, std::move(declaration));
  if (!record)
    return record.takeError();
  return ::fabric::MemoryOperationPortDeclaration{
      {record->endpointInventory().begin(), record->endpointInventory().end()},
      record->resourceContract(),
      {record->operationPatterns().begin(), record->operationPatterns().end()},
      {record->capabilityAlternatives().begin(),
       record->capabilityAlternatives().end()}};
}

llvm::Expected<::fabric::MemoryServiceContractRecord>
localServiceContract(mlir::MLIRContext &context, std::uint64_t capacityBytes,
                     CatalogMemoryDomain domain,
                     const MemoryInterfaceParameters &interface) {
  const std::uint64_t maximumPayloadBits =
      interface.accessDomain.dataPayloadBits / 8 * 8;
  const std::uint64_t maximumBeatCount =
      (maximumPayloadBits + interface.serviceBeatWidthBits - 1) /
      interface.serviceBeatWidthBits;
  auto resources = memoryServiceResourceContract(
      static_cast<std::uint32_t>(maximumBeatCount));
  if (!resources)
    return resources.takeError();
  std::vector<::fabric::MemoryServiceCapabilityDeclaration> capabilities;
  for (bool reads : {true, false}) {
    auto actors = actorDomain(reads);
    if (!actors)
      return actors.takeError();
    auto accesses =
        accessDomain(reads, domain, interface.accessDomain,
                     interface.accessDomain.indexedAddressPayloadBits
                         ? AccessProjectionDomain::All
                         : AccessProjectionDomain::Direct);
    if (!accesses)
      return accesses.takeError();
    capabilities.push_back({std::move(*actors),
                            std::move(*accesses),
                            {0},
                            interface.serviceBeatWidthBits,
                            {::fabric::UsePatternKey(reads ? 0 : 1)},
                            ::fabric::NoMemoryServiceConsistency{}});
  }
  return ::fabric::MemoryServiceContractRecord::create(
      &context, ::fabric::MemoryServiceOwnerKind::Local,
      {{{0, capacityBytes, ::fabric::MemoryServiceRegionBehavior::Storage,
         std::nullopt}},
       std::move(*resources),
       std::move(capabilities)});
}

llvm::Expected<PortType> channelPort(std::uint32_t width,
                                     std::optional<std::uint32_t> tagWidth) {
  return tagWidth ? PortType::taggedBits(width, *tagWidth)
                  : PortType::bits(width);
}

} // namespace

namespace {

llvm::Expected<MemorySpec> makeLocalMemory(LocalMemoryParameters parameters,
                                           CatalogMemoryDomain domain) {
  if (parameters.capacityBytes == 0)
    return invalid("local memory capacity must be positive");
  if (parameters.capacityBytes > (std::uint64_t(1) << 32))
    return invalid("local memory exceeds its 32-bit address capacity");
  const std::uint32_t requiredDataWidth =
      domain == CatalogMemoryDomain::General64 ? 64 : 32;
  if (parameters.interface.accessDomain.dataPayloadBits < requiredDataWidth)
    return invalid("local memory data endpoint cannot carry its scalar floor");
  if (parameters.interface.scalarAddressPayloadBits < 32)
    return invalid("local memory scalar address endpoint is narrower than i32");
  if (parameters.interface.accessDomain.maskPayloadBits == 0)
    return invalid("local memory mask endpoint must be positive");
  if (parameters.interface.serviceBeatWidthBits == 0)
    return invalid("local memory service beat width must be positive");
  if (parameters.interface.accessDomain.indexedAddressPayloadBits &&
      *parameters.interface.accessDomain.indexedAddressPayloadBits == 0)
    return invalid("local memory indexed address endpoint must be positive");
  if (domain == CatalogMemoryDomain::Hybrid32 &&
      parameters.interface.accessDomain.indexedAddressPayloadBits)
    return invalid("Hybrid32 memory has no indexed address endpoint");
  std::optional<std::uint32_t> tagWidth;
  std::optional<std::uint64_t> residentContexts;
  ::fabric::Schedule schedule = ::fabric::Schedule::Spatial;
  if (parameters.temporal) {
    if (parameters.temporal->tagWidth == 0)
      return invalid("temporal local memory requires a positive tag width");
    if (parameters.temporal->residentContextCount == 0)
      return invalid(
          "temporal local memory requires positive resident contexts");
    schedule = ::fabric::Schedule::Temporal;
    tagWidth = parameters.temporal->tagWidth;
    residentContexts = parameters.temporal->residentContextCount;
  }

  std::vector<PortType> inputs;
  std::vector<std::uint32_t> managerOrdinals;
  if (parameters.managerEndpoint) {
    auto byte = PortType::bits(8);
    if (!byte)
      return byte.takeError();
    auto manager = PortType::memory({PortType::kDynamicExtent}, *byte);
    if (!manager)
      return manager.takeError();
    inputs.push_back(std::move(*manager));
    managerOrdinals.push_back(0);
  }
  const MemoryEndpointLayout layout = endpointLayout(parameters.interface);
  for (std::uint32_t width : layout.inputWidths) {
    auto type = channelPort(width, tagWidth);
    if (!type)
      return type.takeError();
    inputs.push_back(std::move(*type));
  }
  std::vector<PortType> outputs;
  for (std::uint32_t width : layout.outputWidths) {
    auto type = channelPort(width, tagWidth);
    if (!type)
      return type.takeError();
    outputs.push_back(std::move(*type));
  }

  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const auto endpoints = endpointInventory(layout, tagWidth);
  std::vector<::fabric::MemoryOperationPortDeclaration> operationPorts;
  for (bool reads : {true, false}) {
    auto port = operationPort(context, schedule, endpoints, layout, reads,
                              domain, parameters.interface.accessDomain);
    if (!port)
      return port.takeError();
    operationPorts.push_back(std::move(*port));
  }
  auto serviceContract = localServiceContract(context, parameters.capacityBytes,
                                              domain, parameters.interface);
  if (!serviceContract)
    return serviceContract.takeError();
  auto service = LocalMemoryServiceSpec::create(parameters.capacityBytes,
                                                *serviceContract);
  if (!service)
    return service.takeError();

  ::fabric::MemoryConnectivityDeclaration connectivityDeclaration;
  for (std::size_t port = 0; port != operationPorts.size(); ++port) {
    ::fabric::MemoryOperationPortDispatchDeclaration dispatch;
    dispatch.capabilityTargetDomains.resize(
        operationPorts[port].capabilityAlternatives.size());
    for (auto &targets : dispatch.capabilityTargetDomains) {
      targets.push_back(::fabric::MemoryDispatchTarget(
          std::in_place_type<::fabric::LocalMemoryDispatchTarget>));
      if (parameters.managerEndpoint)
        targets.push_back(::fabric::MemoryDispatchTarget(
            std::in_place_type<::fabric::ManagerMemoryDispatchTarget>,
            ::fabric::ManagerMemoryDispatchTarget{0}));
    }
    connectivityDeclaration.operationPorts.push_back(std::move(dispatch));
  }
  auto connectivity =
      MemoryConnectivitySpec::create(std::move(connectivityDeclaration));
  if (!connectivity)
    return connectivity.takeError();

  std::optional<MemoryEngineSpec> engine;
  if (residentContexts)
    engine = MemoryEngineSpec::temporal(*residentContexts,
                                        std::move(operationPorts));
  else
    engine = MemoryEngineSpec::spatial(std::move(operationPorts));
  return MemorySpec::create(std::move(inputs), std::move(outputs),
                            std::move(managerOrdinals), {}, std::move(engine),
                            std::move(*service), std::move(*connectivity));
}

llvm::Expected<SystemMemorySpec>
makeSystemMemory(SystemMemoryParameters parameters,
                 loom::fabric::ServiceRateContractRecord serviceRate,
                 CatalogMemoryDomain domain) {
  if (parameters.capacityBytes == 0)
    return invalid("System memory capacity must be positive");
  const std::uint32_t requiredDataWidth =
      domain == CatalogMemoryDomain::General64 ? 64 : 32;
  if (parameters.accessDomain.dataPayloadBits < requiredDataWidth)
    return invalid("System memory access domain cannot carry its scalar floor");
  if (parameters.accessDomain.maskPayloadBits == 0)
    return invalid("System memory mask domain must be positive");
  if (parameters.serviceBeatWidthBits == 0)
    return invalid("System memory service beat width must be positive");
  if (parameters.accessDomain.indexedAddressPayloadBits &&
      *parameters.accessDomain.indexedAddressPayloadBits == 0)
    return invalid("System memory indexed address domain must be positive");
  if (domain == CatalogMemoryDomain::Hybrid32 &&
      parameters.accessDomain.indexedAddressPayloadBits)
    return invalid("Hybrid32 System memory has no indexed address domain");
  auto endAddress = llvm::checkedAddUnsigned(parameters.addressBaseBytes,
                                             parameters.capacityBytes);
  if (!endAddress)
    return invalid("System memory address range overflows u64");
  const std::uint64_t lastAddress = *endAddress - 1;

  const std::uint64_t maximumPayloadBits =
      parameters.accessDomain.dataPayloadBits / 8 * 8;
  const std::uint64_t maximumBeatCount =
      (maximumPayloadBits + parameters.serviceBeatWidthBits - 1) /
      parameters.serviceBeatWidthBits;
  auto resources = memoryServiceResourceContract(
      static_cast<std::uint32_t>(maximumBeatCount));
  if (!resources)
    return resources.takeError();
  std::vector<::fabric::MemoryServiceCapabilityDeclaration> serviceCapabilities;
  std::vector<loom::fabric::CanonicalServiceCapabilityRecord>
      endpointCapabilities;
  for (bool reads : {true, false}) {
    auto serviceActors = actorDomain(reads);
    if (!serviceActors)
      return serviceActors.takeError();
    auto serviceAccesses =
        accessDomain(reads, domain, parameters.accessDomain,
                     parameters.accessDomain.indexedAddressPayloadBits
                         ? AccessProjectionDomain::All
                         : AccessProjectionDomain::Direct);
    if (!serviceAccesses)
      return serviceAccesses.takeError();
    serviceCapabilities.push_back({std::move(*serviceActors),
                                   std::move(*serviceAccesses),
                                   {0},
                                   parameters.serviceBeatWidthBits,
                                   {::fabric::UsePatternKey(reads ? 0 : 1)},
                                   ::fabric::NoMemoryServiceConsistency{}});

    auto endpointActors = actorDomain(reads);
    if (!endpointActors)
      return endpointActors.takeError();
    auto endpointAccesses =
        accessDomain(reads, domain, parameters.accessDomain,
                     parameters.accessDomain.indexedAddressPayloadBits
                         ? AccessProjectionDomain::All
                         : AccessProjectionDomain::Direct);
    if (!endpointAccesses)
      return endpointAccesses.takeError();
    auto addressDomain = ::fabric::UnsignedDomain::fromCanonical(
        {{parameters.addressBaseBytes, lastAddress}});
    if (!addressDomain)
      return addressDomain.takeError();
    auto domain = loom::fabric::AddressedMemoryCapabilityDomain::create(
        std::move(*endpointActors), std::move(*endpointAccesses),
        std::move(*addressDomain), parameters.serviceBeatWidthBits,
        std::nullopt);
    if (!domain)
      return domain.takeError();
    auto capability = loom::fabric::CanonicalServiceCapabilityRecord::create(
        reads ? ::dataflow::semantics::ServiceKind::MemoryRead
              : ::dataflow::semantics::ServiceKind::MemoryWrite,
        loom::fabric::CanonicalServiceEndpointRole::Serve, std::move(*domain),
        serviceRate);
    if (!capability)
      return capability.takeError();
    endpointCapabilities.push_back(std::move(*capability));
  }

  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  auto contract = ::fabric::MemoryServiceContractRecord::create(
      &context, ::fabric::MemoryServiceOwnerKind::System,
      {{{parameters.addressBaseBytes, parameters.capacityBytes,
         ::fabric::MemoryServiceRegionBehavior::Storage, std::nullopt}},
       std::move(*resources),
       std::move(serviceCapabilities)});
  if (!contract)
    return contract.takeError();
  auto capabilities = loom::fabric::CanonicalServiceCapabilitySet::create(
      std::move(endpointCapabilities));
  if (!capabilities)
    return capabilities.takeError();
  return SystemMemorySpec{std::move(*contract), std::move(*capabilities)};
}

} // namespace

llvm::Expected<MemorySpec>
makeHybrid32LocalMemory(LocalMemoryParameters parameters) {
  return makeLocalMemory(std::move(parameters), CatalogMemoryDomain::Hybrid32);
}

llvm::Expected<MemorySpec>
makeGeneral64LocalMemory(LocalMemoryParameters parameters) {
  return makeLocalMemory(std::move(parameters), CatalogMemoryDomain::General64);
}

llvm::Expected<SystemMemorySpec>
makeHybrid32SystemMemory(SystemMemoryParameters parameters,
                         loom::fabric::ServiceRateContractRecord serviceRate) {
  return makeSystemMemory(std::move(parameters), std::move(serviceRate),
                          CatalogMemoryDomain::Hybrid32);
}

llvm::Expected<SystemMemorySpec>
makeGeneral64SystemMemory(SystemMemoryParameters parameters,
                          loom::fabric::ServiceRateContractRecord serviceRate) {
  return makeSystemMemory(std::move(parameters), std::move(serviceRate),
                          CatalogMemoryDomain::General64);
}

} // namespace loom::adg
