#include "ADG/MemoryLibrary.h"

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

#include <optional>
#include <utility>
#include <vector>

namespace loom::adg {
namespace {

using ::dataflow::OperationSchemaId;
using ::dataflow::semantics::MemoryAccessForm;
using ::dataflow::semantics::MemoryMaskForm;
using ::dataflow::semantics::ServiceValueRole;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "adg_memory_library_invalid: " + message);
}

llvm::Expected<::fabric::UnsignedDomain> singleton(std::uint64_t value) {
  return ::fabric::UnsignedDomain::fromCanonical({{value, value}});
}

llvm::Expected<::fabric::UnsignedDomain> scalarElementWidths() {
  return ::fabric::UnsignedDomain::fromCanonical({{8, 8}, {16, 16}, {32, 32}});
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

llvm::Expected<::fabric::MemoryAccessClass> elementAccess(bool reads) {
  auto widths = scalarElementWidths();
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
      std::move(*alignments), std::move(*read), std::move(*write));
}

llvm::Expected<::fabric::MemoryAccessClass> vectorAccess(bool reads) {
  auto widths = singleton(32);
  if (!widths)
    return widths.takeError();
  auto lanes = singleton(4);
  if (!lanes)
    return lanes.takeError();
  auto alignments = allAlignments();
  if (!alignments)
    return alignments.takeError();
  auto read = enumDomain<::fabric::ReadSubwordSemantics>(
      {reads ? ::fabric::ReadSubwordSemantics::Exact
             : ::fabric::ReadSubwordSemantics::NotApplicable});
  if (!read)
    return read.takeError();
  auto write = enumDomain<::fabric::WriteSubwordSemantics>(
      {reads ? ::fabric::WriteSubwordSemantics::NotApplicable
             : ::fabric::WriteSubwordSemantics::Exact});
  if (!write)
    return write.takeError();
  return ::fabric::MemoryAccessClass::create(
      MemoryAccessForm::Contiguous, std::move(*widths), std::move(*lanes),
      {{MemoryMaskForm::Absent, ::fabric::InactiveLaneSemantics::NotApplicable},
       {MemoryMaskForm::Dynamic,
        reads ? ::fabric::InactiveLaneSemantics::SuppressAndZeroFill
              : ::fabric::InactiveLaneSemantics::Suppress}},
      std::move(*alignments), std::move(*read), std::move(*write));
}

llvm::Expected<::fabric::ParameterizedMemoryAccessDomain>
accessDomain(bool reads) {
  auto element = elementAccess(reads);
  if (!element)
    return element.takeError();
  auto vector = vectorAccess(reads);
  if (!vector)
    return vector.takeError();
  return ::fabric::ParameterizedMemoryAccessDomain::create({*element, *vector});
}

llvm::Expected<::fabric::MemoryActorContractDomain> actorDomain(bool reads) {
  ::fabric::MemoryActorContractClause plain =
      ::fabric::LoadStorePlainContractClause{{false}};
  return ::fabric::MemoryActorContractDomain::create(
      reads ? OperationSchemaId::DataflowLoad
            : OperationSchemaId::DataflowStore,
      {plain});
}

llvm::Expected<::fabric::ResourceContract> operationPortResourceContract() {
  ::fabric::ResourceContractDeclaration declaration;
  declaration.states = {
      {::fabric::StateKey(0),
       {{::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(1),
         ::fabric::CapacityUnits(0)}}}};
  declaration.requesters = {::fabric::RequesterKey(0)};
  declaration.eligibilityCount = 1;
  declaration.eventCount = 2;
  declaration.timingContracts = {{::fabric::TimingContractKey(0), {0, 1}}};
  declaration.usePatterns = {
      {::fabric::UsePatternKey(0),
       ::fabric::RequesterKey(0),
       ::fabric::EligibilityKey(0),
       ::fabric::EventKey(0),
       ::fabric::EventKey(1),
       std::nullopt,
       ::fabric::TimingContractKey(0),
       {{::fabric::ClaimKey(0), ::fabric::StateKey(0),
         ::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(1)}},
       {{{::fabric::ClaimKey(0)}}}}};
  return ::fabric::ResourceContract::create(declaration);
}

llvm::Expected<::fabric::ResourceContract> memoryServiceResourceContract() {
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
         {{{::fabric::ClaimKey(0)}}}});
  declaration.grantPolicy = ::fabric::RoundRobinDeclaration{
      {::fabric::RequesterKey(0), ::fabric::RequesterKey(1)},
      ::fabric::RequesterKey(0)};
  return ::fabric::ResourceContract::create(declaration);
}

std::vector<::fabric::MemoryTransportEndpointDescriptor>
endpointInventory(std::optional<std::uint32_t> tagWidth) {
  using loom::fabric::FabricPortDirection;
  std::vector<::fabric::MemoryTransportEndpointDescriptor> endpoints;
  for (std::uint32_t width : {64u, 4u, 0u, 64u, 128u, 4u, 0u})
    endpoints.push_back({FabricPortDirection::Input, width, tagWidth});
  for (std::uint32_t width : {128u, 0u, 0u})
    endpoints.push_back({FabricPortDirection::Output, width, tagWidth});
  return endpoints;
}

llvm::Expected<::fabric::MemoryOperationPortDeclaration> operationPort(
    mlir::MLIRContext &context, ::fabric::Schedule schedule,
    llvm::ArrayRef<::fabric::MemoryTransportEndpointDescriptor> endpoints,
    bool reads) {
  auto resources = operationPortResourceContract();
  if (!resources)
    return resources.takeError();
  auto actors = actorDomain(reads);
  if (!actors)
    return actors.takeError();
  auto accesses = accessDomain(reads);
  if (!accesses)
    return accesses.takeError();

  ::fabric::MemoryCapabilityAlternativeRecord alternative{
      std::move(*actors),
      reads
          ? std::vector<
                ::fabric::
                    MemoryRoleEndpointBindingRecord>{{ServiceValueRole::Address,
                                                      0},
                                                     {ServiceValueRole::Data,
                                                      7},
                                                     {ServiceValueRole::Mask,
                                                      1},
                                                     {ServiceValueRole::Control,
                                                      2},
                                                     {ServiceValueRole::
                                                          Completion,
                                                      8}}
          : std::vector<
                ::fabric::
                    MemoryRoleEndpointBindingRecord>{{ServiceValueRole::Address,
                                                      3},
                                                     {ServiceValueRole::Data,
                                                      4},
                                                     {ServiceValueRole::Mask,
                                                      5},
                                                     {ServiceValueRole::Control,
                                                      6},
                                                     {ServiceValueRole::
                                                          Completion,
                                                      9}},
      std::move(*accesses),
      {::fabric::UsePatternKey(0)}};
  ::fabric::MemoryOperationPortDeclaration declaration{
      reads ? std::vector<std::uint64_t>{0, 1, 2, 7, 8}
            : std::vector<std::uint64_t>{3, 4, 5, 6, 9},
      std::move(*resources),
      {{::fabric::MemoryPortTransactionProjection::Direct}},
      {std::move(alternative)}};
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
localServiceContract(mlir::MLIRContext &context, std::uint64_t capacityBytes) {
  auto resources = memoryServiceResourceContract();
  if (!resources)
    return resources.takeError();
  std::vector<::fabric::MemoryServiceCapabilityDeclaration> capabilities;
  for (bool reads : {true, false}) {
    auto actors = actorDomain(reads);
    if (!actors)
      return actors.takeError();
    auto accesses = accessDomain(reads);
    if (!accesses)
      return accesses.takeError();
    capabilities.push_back({std::move(*actors),
                            std::move(*accesses),
                            {0},
                            128,
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

llvm::Expected<MemorySpec>
makeHybrid32LocalMemory(Hybrid32LocalMemoryParameters parameters) {
  if (parameters.capacityBytes == 0)
    return invalid("local memory capacity must be positive");
  if (parameters.capacityBytes > (std::uint64_t(1) << 32))
    return invalid("local memory exceeds its 32-bit address capacity");
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
  for (std::uint32_t width : {64u, 4u, 0u, 64u, 128u, 4u, 0u}) {
    auto type = channelPort(width, tagWidth);
    if (!type)
      return type.takeError();
    inputs.push_back(std::move(*type));
  }
  std::vector<PortType> outputs;
  for (std::uint32_t width : {128u, 0u, 0u}) {
    auto type = channelPort(width, tagWidth);
    if (!type)
      return type.takeError();
    outputs.push_back(std::move(*type));
  }

  mlir::MLIRContext context;
  const auto endpoints = endpointInventory(tagWidth);
  std::vector<::fabric::MemoryOperationPortDeclaration> operationPorts;
  for (bool reads : {true, false}) {
    auto port = operationPort(context, schedule, endpoints, reads);
    if (!port)
      return port.takeError();
    operationPorts.push_back(std::move(*port));
  }
  auto serviceContract =
      localServiceContract(context, parameters.capacityBytes);
  if (!serviceContract)
    return serviceContract.takeError();
  auto service = LocalMemoryServiceSpec::create(parameters.capacityBytes,
                                                *serviceContract);
  if (!service)
    return service.takeError();

  ::fabric::MemoryConnectivityDeclaration connectivityDeclaration;
  for (std::size_t port = 0; port != operationPorts.size(); ++port) {
    ::fabric::MemoryOperationPortDispatchDeclaration dispatch;
    dispatch.capabilityTargetDomains = {{::fabric::MemoryDispatchTarget(
        std::in_place_type<::fabric::LocalMemoryDispatchTarget>)}};
    if (parameters.managerEndpoint)
      dispatch.capabilityTargetDomains.front().push_back(
          ::fabric::MemoryDispatchTarget(
              std::in_place_type<::fabric::ManagerMemoryDispatchTarget>,
              ::fabric::ManagerMemoryDispatchTarget{0}));
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
  return MemorySpec::create(
      std::move(inputs), std::move(outputs), std::move(managerOrdinals), {},
      std::move(engine), std::move(*service), std::move(*connectivity));
}

llvm::Expected<Hybrid32SystemMemorySpec>
makeHybrid32SystemMemory(Hybrid32SystemMemoryParameters parameters,
                         loom::fabric::ServiceRateContractRecord serviceRate) {
  if (parameters.capacityBytes == 0)
    return invalid("System memory capacity must be positive");
  auto endAddress = llvm::checkedAddUnsigned(parameters.addressBaseBytes,
                                             parameters.capacityBytes);
  if (!endAddress)
    return invalid("System memory address range overflows u64");
  const std::uint64_t lastAddress = *endAddress - 1;

  auto resources = memoryServiceResourceContract();
  if (!resources)
    return resources.takeError();
  std::vector<::fabric::MemoryServiceCapabilityDeclaration> serviceCapabilities;
  std::vector<loom::fabric::CanonicalServiceCapabilityRecord>
      endpointCapabilities;
  for (bool reads : {true, false}) {
    auto serviceActors = actorDomain(reads);
    if (!serviceActors)
      return serviceActors.takeError();
    auto serviceAccesses = accessDomain(reads);
    if (!serviceAccesses)
      return serviceAccesses.takeError();
    serviceCapabilities.push_back({std::move(*serviceActors),
                                   std::move(*serviceAccesses),
                                   {0},
                                   128,
                                   {::fabric::UsePatternKey(reads ? 0 : 1)},
                                   ::fabric::NoMemoryServiceConsistency{}});

    auto endpointActors = actorDomain(reads);
    if (!endpointActors)
      return endpointActors.takeError();
    auto endpointAccesses = accessDomain(reads);
    if (!endpointAccesses)
      return endpointAccesses.takeError();
    auto addressDomain = ::fabric::UnsignedDomain::fromCanonical(
        {{parameters.addressBaseBytes, lastAddress}});
    if (!addressDomain)
      return addressDomain.takeError();
    auto domain = loom::fabric::AddressedMemoryCapabilityDomain::create(
        std::move(*endpointActors), std::move(*endpointAccesses),
        std::move(*addressDomain), 128, std::nullopt);
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

  mlir::MLIRContext context;
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
  return Hybrid32SystemMemorySpec{std::move(*contract),
                                  std::move(*capabilities)};
}

} // namespace loom::adg
