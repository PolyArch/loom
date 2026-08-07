#include "Fabric/IR/SystemServiceContract.h"

#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <limits>
#include <set>
#include <system_error>
#include <utility>

using namespace loom::fabric;

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "system_service_contract_invalid: " + message);
}

bool isAllowedOwner(const FabricInventoryOwnerRef &owner) {
  const auto &payload = owner.payload;
  if (std::holds_alternative<HostCoreOccurrenceRef>(payload) ||
      std::holds_alternative<AccCoreOccurrenceRef>(payload) ||
      std::holds_alternative<SystemServiceTransformRef>(payload) ||
      std::holds_alternative<ExternalBoundaryRef>(payload))
    return true;
  const auto *memory = std::get_if<FabricMemoryServiceRef>(&payload);
  return memory && memory->kind() == FabricMemoryServiceKind::System;
}

llvm::Expected<std::vector<std::uint8_t>> canonicalTypeBytes(mlir::Type type) {
  auto encoded = dataflow::encodeCanonicalType(type);
  if (!encoded)
    return encoded.takeError();
  return std::vector<std::uint8_t>(encoded->bytes().begin(),
                                   encoded->bytes().end());
}

bool requiresConsistency(
    const ::fabric::MemoryActorContractDomain &actorContracts) {
  return llvm::any_of(actorContracts.clauses(), [](const auto &clause) {
    return !std::holds_alternative<::fabric::LoadStorePlainContractClause>(
        clause);
  });
}

bool knownRole(CanonicalServiceEndpointRole role) {
  switch (role) {
  case CanonicalServiceEndpointRole::Initiate:
  case CanonicalServiceEndpointRole::Serve:
    return true;
  }
  return false;
}

llvm::Expected<std::vector<std::uint8_t>>
endpointBytes(const FabricMemoryEndpointRef &endpoint) {
  return canonicalFabricBytes(endpoint);
}

llvm::Error
rejectDuplicateEndpoints(llvm::ArrayRef<FabricMemoryEndpointRef> endpoints,
                         llvm::StringRef field) {
  std::vector<std::vector<std::uint8_t>> keys;
  keys.reserve(endpoints.size());
  for (const FabricMemoryEndpointRef &endpoint : endpoints) {
    auto key = endpointBytes(endpoint);
    if (!key)
      return key.takeError();
    keys.push_back(std::move(*key));
  }
  llvm::sort(keys);
  if (std::adjacent_find(keys.begin(), keys.end()) != keys.end())
    return invalid(field + " contains a duplicate endpoint");
  return llvm::Error::success();
}

std::vector<std::uint8_t>
correspondenceKey(const CoherentMemoryRegionCorrespondence &correspondence) {
  const std::vector<std::uint8_t> input =
      canonicalFabricBytes(correspondence.input);
  const std::vector<std::uint8_t> output =
      canonicalFabricBytes(correspondence.output);
  std::vector<std::uint8_t> key;
  for (int shift = 56; shift >= 0; shift -= 8)
    key.push_back(static_cast<std::uint8_t>(input.size() >> shift));
  key.insert(key.end(), input.begin(), input.end());
  for (int shift = 56; shift >= 0; shift -= 8)
    key.push_back(static_cast<std::uint8_t>(output.size() >> shift));
  key.insert(key.end(), output.begin(), output.end());
  return key;
}

} // namespace

llvm::Expected<SystemServiceEndpointOwnerRef>
SystemServiceEndpointOwnerRef::create(FabricInventoryOwnerRef owner) {
  if (!isAllowedOwner(owner))
    return invalid("service endpoint owner is outside the closed System owner "
                   "domain");
  return SystemServiceEndpointOwnerRef(std::move(owner));
}

llvm::Expected<ServiceRateContractRecord> ServiceRateContractRecord::create(
    ClockDomainRef rateClock, std::uint64_t operationsPerWindow,
    std::uint64_t windowTicks, std::uint64_t maxOutstanding,
    ServiceProgress progress) {
  if (operationsPerWindow == 0 || windowTicks == 0)
    return invalid("service rate must be positive");
  if (maxOutstanding == 0)
    return invalid("service outstanding capacity must be positive");
  if (const auto *bounded = std::get_if<::fabric::BoundedCompletion>(&progress))
    if (bounded->maxIssueToRetireTicks == 0)
      return invalid("bounded service progress must be positive");
  return ServiceRateContractRecord(std::move(rateClock), operationsPerWindow,
                                   windowTicks, maxOutstanding,
                                   std::move(progress));
}

llvm::Expected<MessageTransferCapabilityDomain>
MessageTransferCapabilityDomain::create(
    llvm::ArrayRef<mlir::Type> payloadTypes) {
  if (payloadTypes.empty())
    return invalid("message payload domain must not be empty");
  std::vector<std::pair<std::vector<std::uint8_t>, mlir::Type>> ordered;
  ordered.reserve(payloadTypes.size());
  for (mlir::Type type : payloadTypes) {
    auto bytes = canonicalTypeBytes(type);
    if (!bytes)
      return bytes.takeError();
    ordered.emplace_back(std::move(*bytes), type);
  }
  llvm::sort(ordered, [](const auto &left, const auto &right) {
    return left.first < right.first;
  });
  std::vector<mlir::Type> normalized;
  normalized.reserve(ordered.size());
  for (const auto &[bytes, type] : ordered) {
    if (!normalized.empty()) {
      auto previous = canonicalTypeBytes(normalized.back());
      if (!previous)
        return previous.takeError();
      if (*previous == bytes)
        continue;
    }
    normalized.push_back(type);
  }
  return MessageTransferCapabilityDomain(std::move(normalized));
}

llvm::Expected<MessageTransferCapabilityDomain>
MessageTransferCapabilityDomain::fromCanonical(
    llvm::ArrayRef<mlir::Type> payloadTypes) {
  if (payloadTypes.empty())
    return invalid("message payload domain must not be empty");
  std::vector<std::uint8_t> previous;
  bool hasPrevious = false;
  for (mlir::Type type : payloadTypes) {
    auto bytes = canonicalTypeBytes(type);
    if (!bytes)
      return bytes.takeError();
    if (hasPrevious && previous >= *bytes)
      return invalid("message payload domain is not sorted and unique");
    previous = std::move(*bytes);
    hasPrevious = true;
  }
  return MessageTransferCapabilityDomain(
      std::vector<mlir::Type>(payloadTypes.begin(), payloadTypes.end()));
}

llvm::Expected<AddressedMemoryCapabilityDomain>
AddressedMemoryCapabilityDomain::create(
    ::fabric::MemoryActorContractDomain actorContracts,
    ::fabric::ParameterizedMemoryAccessDomain accesses,
    ::fabric::UnsignedDomain addressBytes, std::uint64_t serviceBeatWidthBits,
    std::optional<MemoryConsistencyDomainRef> consistencyDomain) {
  if (serviceBeatWidthBits == 0)
    return invalid("addressed service beat width must be positive");
  return AddressedMemoryCapabilityDomain(
      std::move(actorContracts), std::move(accesses), std::move(addressBytes),
      serviceBeatWidthBits, std::move(consistencyDomain));
}

llvm::Expected<FenceCapabilityDomain> FenceCapabilityDomain::create(
    ::fabric::MemoryActorContractDomain actorContracts,
    MemoryConsistencyDomainRef consistencyDomain) {
  auto kind =
      dataflow::semantics::getMemoryServiceKind(actorContracts.actorSchema());
  if (!kind)
    return kind.takeError();
  if (*kind != dataflow::semantics::ServiceKind::MemoryFence)
    return invalid("fence capability uses a non-fence actor domain");
  return FenceCapabilityDomain(std::move(actorContracts),
                               std::move(consistencyDomain));
}

llvm::Expected<CanonicalServiceCapabilityRecord>
CanonicalServiceCapabilityRecord::create(
    dataflow::semantics::ServiceKind kind, CanonicalServiceEndpointRole role,
    CanonicalServiceCapabilityDomain domain, ServiceRateContractRecord rate) {
  if (!knownRole(role))
    return invalid("unknown service endpoint role");

  if (const auto *message =
          std::get_if<MessageTransferCapabilityDomain>(&domain)) {
    (void)message;
    if (kind != dataflow::semantics::ServiceKind::MessageTransfer)
      return invalid("message payload domain is bound to a non-message kind");
  } else if (const auto *addressed =
                 std::get_if<AddressedMemoryCapabilityDomain>(&domain)) {
    if (kind == dataflow::semantics::ServiceKind::MessageTransfer ||
        kind == dataflow::semantics::ServiceKind::MemoryFence)
      return invalid("addressed domain is bound to a non-addressed kind");
    auto actorKind = dataflow::semantics::getMemoryServiceKind(
        addressed->actorContracts().actorSchema());
    if (!actorKind)
      return actorKind.takeError();
    if (*actorKind != kind)
      return invalid("service kind does not match its actor-contract domain");
    const bool required = requiresConsistency(addressed->actorContracts());
    if (required != addressed->consistencyDomain().has_value())
      return invalid("consistency domain presence does not match the accepted "
                     "actor contracts");
  } else {
    const auto &fence = std::get<FenceCapabilityDomain>(domain);
    (void)fence;
    if (kind != dataflow::semantics::ServiceKind::MemoryFence)
      return invalid("fence domain is bound to a non-fence kind");
  }
  return CanonicalServiceCapabilityRecord(kind, role, std::move(domain),
                                          std::move(rate));
}

llvm::Expected<ServiceLegCarrierAttachmentRecord>
ServiceLegCarrierAttachmentRecord::create(
    FabricMemoryEndpointRef endpoint, dataflow::semantics::ServiceKind kind,
    dataflow::StructuralOrdinal legOrdinal,
    std::vector<FabricTransportEndpointRef> carriers) {
  if (kind == dataflow::semantics::ServiceKind::MessageTransfer)
    return invalid("MessageTransfer does not use memory service leg carrier "
                   "attachments");
  if (carriers.empty())
    return invalid("service leg carrier set must not be empty");
  llvm::sort(carriers, [](const FabricTransportEndpointRef &left,
                          const FabricTransportEndpointRef &right) {
    return canonicalFabricBytes(left) < canonicalFabricBytes(right);
  });
  carriers.erase(std::unique(carriers.begin(), carriers.end()), carriers.end());
  return ServiceLegCarrierAttachmentRecord(std::move(endpoint), kind,
                                           legOrdinal, std::move(carriers));
}

llvm::Expected<ServiceLegCarrierAttachmentRecord>
ServiceLegCarrierAttachmentRecord::fromCanonical(
    FabricMemoryEndpointRef endpoint, dataflow::semantics::ServiceKind kind,
    dataflow::StructuralOrdinal legOrdinal,
    std::vector<FabricTransportEndpointRef> carriers) {
  if (kind == dataflow::semantics::ServiceKind::MessageTransfer)
    return invalid("MessageTransfer does not use memory service leg carrier "
                   "attachments");
  if (carriers.empty())
    return invalid("service leg carrier set must not be empty");
  for (std::size_t index = 1; index < carriers.size(); ++index)
    if (canonicalFabricBytes(carriers[index - 1]) >=
        canonicalFabricBytes(carriers[index]))
      return invalid("service leg carrier set is not sorted and unique");
  return ServiceLegCarrierAttachmentRecord(std::move(endpoint), kind,
                                           legOrdinal, std::move(carriers));
}

llvm::Expected<SystemServiceTransformRecord>
SystemServiceTransformRecord::create(
    std::vector<FabricMemoryEndpointRef> inputs,
    std::vector<FabricMemoryEndpointRef> outputs,
    ServiceTransformContract contract) {
  if (inputs.empty() || outputs.empty())
    return invalid("service transform endpoints must not be empty");
  if (llvm::Error error = rejectDuplicateEndpoints(inputs, "transform inputs"))
    return std::move(error);
  if (llvm::Error error =
          rejectDuplicateEndpoints(outputs, "transform outputs"))
    return std::move(error);

  if (auto *offset = std::get_if<AddressOffsetTransform>(&contract)) {
    if (inputs.size() != 1 || outputs.size() != 1)
      return invalid("AddressOffset requires one input and one output");
    if (offset->addressWidth == 0 || offset->addressWidth > 64)
      return invalid("AddressOffset address width must be in [1, 64]");
    if (offset->signedOffset == 0)
      return invalid("identity AddressOffset must be a direct connection");
  } else if (auto *mask = std::get_if<AddressMaskXorTransform>(&contract)) {
    if (inputs.size() != 1 || outputs.size() != 1)
      return invalid("AddressMaskXor requires one input and one output");
    if (mask->addressWidth == 0 || mask->addressWidth > 64)
      return invalid("AddressMaskXor address width must be in [1, 64]");
    const std::uint64_t widthMask =
        mask->addressWidth == 64 ? std::numeric_limits<std::uint64_t>::max()
                                 : (std::uint64_t{1} << mask->addressWidth) - 1;
    if ((mask->andMask | mask->xorMask) & ~widthMask)
      return invalid("AddressMaskXor mask exceeds its address width");
    if (mask->andMask == widthMask && mask->xorMask == 0)
      return invalid("identity AddressMaskXor must be a direct connection");
  } else if (auto *interleave =
                 std::get_if<StaticInterleaveTransform>(&contract)) {
    if (inputs.size() != 1 || outputs.size() != interleave->outputCount)
      return invalid("StaticInterleave endpoint arity does not match its "
                     "output count");
    if (interleave->granuleBytes == 0 || interleave->outputCount < 2)
      return invalid("StaticInterleave requires a positive granule and at "
                     "least two outputs");
  } else {
    auto &coherent = std::get<CoherentMemoryTransform>(contract);
    if (coherent.regions.empty())
      return invalid("CoherentMemory region correspondence must not be empty");
    llvm::sort(coherent.regions, [](const auto &left, const auto &right) {
      return correspondenceKey(left) < correspondenceKey(right);
    });
    for (std::size_t index = 1; index < coherent.regions.size(); ++index)
      if (coherent.regions[index - 1] == coherent.regions[index])
        return invalid("CoherentMemory contains a duplicate region "
                       "correspondence");
    std::set<std::vector<std::uint8_t>> inputs;
    std::set<std::vector<std::uint8_t>> outputs;
    for (const CoherentMemoryRegionCorrespondence &region : coherent.regions) {
      if (!inputs.insert(canonicalFabricBytes(region.input)).second)
        return invalid("CoherentMemory repeats an input region");
      if (!outputs.insert(canonicalFabricBytes(region.output)).second)
        return invalid("CoherentMemory repeats an output region");
    }
  }
  return SystemServiceTransformRecord(std::move(inputs), std::move(outputs),
                                      std::move(contract));
}
