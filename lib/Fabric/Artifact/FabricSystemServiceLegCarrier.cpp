#include "FabricSystemServiceLegCarrier.h"

#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/SystemServiceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;

namespace {

struct AttachmentKey {
  std::vector<std::uint8_t> endpoint;
  dataflow::semantics::ServiceKind kind;
  dataflow::StructuralOrdinal legOrdinal;

  friend bool operator<(const AttachmentKey &left, const AttachmentKey &right) {
    return std::tie(left.endpoint, left.kind, left.legOrdinal) <
           std::tie(right.endpoint, right.kind, right.legOrdinal);
  }

  friend bool operator==(const AttachmentKey &left,
                         const AttachmentKey &right) {
    return left.endpoint == right.endpoint && left.kind == right.kind &&
           left.legOrdinal == right.legOrdinal;
  }
};

struct AttachmentGroup {
  ::fabric::SystemServiceLegCarrierAttachmentOp representative;
  loom::fabric::FabricMemoryEndpointRef endpoint;
  dataflow::semantics::ServiceKind kind;
  dataflow::StructuralOrdinal legOrdinal;
  std::vector<loom::fabric::FabricTransportEndpointRef> carriers;
  llvm::SmallVector<Operation *> duplicates;
};

struct MemoryEndpointContext {
  const loom::fabric::CanonicalServiceCapabilitySet *capabilities;
  loom::fabric::CanonicalServiceEndpointRole effectiveRole;
};

AttachmentKey
attachmentKey(const loom::fabric::ServiceLegCarrierAttachmentRecord &record) {
  return {loom::fabric::canonicalFabricBytes(record.endpoint()), record.kind(),
          record.legOrdinal()};
}

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

llvm::Expected<MemoryEndpointContext> resolveMemoryEndpointContext(
    const loom::fabric::FabricSystemRootView &system,
    const loom::fabric::FabricMemoryEndpointRef &endpoint) {
  using namespace loom::fabric;
  const FabricArtifactView &fabric = system.artifact();
  if (endpoint.owner.kind() ==
      FabricMemoryEndpointOwnerKind::SystemServiceEndpoint) {
    if (endpoint.ordinal != 0)
      return invalid("service-leg System endpoint ordinal is not zero");
    const auto serviceEndpoint =
        std::get<SystemServiceEndpointRef>(endpoint.owner.payload);
    const CanonicalServiceCapabilitySet *capabilities =
        system.serviceEndpointCapabilities(serviceEndpoint);
    if (!capabilities ||
        capabilities->plane() != CanonicalServiceEndpointPlane::Memory)
      return invalid(
          "service-leg attachment endpoint has no memory capability set");
    return MemoryEndpointContext{capabilities, capabilities->role()};
  }

  if (endpoint.owner.kind() !=
      FabricMemoryEndpointOwnerKind::SpatialCoreOccurrence)
    return invalid("service-leg attachment endpoint is not an admitted "
                   "memory attachment pair member");

  const FabricSpatialAttachmentRecordView *binding = nullptr;
  for (const FabricSpatialAttachmentRecordView &candidate :
       system.spatialAttachments()) {
    const FabricMemoryEndpointRef *memory = candidate.spatialEndpoint.memory();
    if (!memory || *memory != endpoint)
      continue;
    if (binding)
      return invalid(
          "service-leg occurrence endpoint has multiple memory attachments");
    binding = &candidate;
  }
  if (!binding || !binding->serviceEndpoint)
    return invalid(
        "service-leg occurrence endpoint has no memory spatial attachment");

  const CanonicalServiceCapabilitySet *capabilities =
      system.serviceEndpointCapabilities(*binding->serviceEndpoint);
  if (!capabilities ||
      capabilities->plane() != CanonicalServiceEndpointPlane::Memory)
    return invalid(
        "service-leg occurrence binding has no memory capability authority");
  const std::optional<FabricMemoryEndpointRole> occurrenceRole =
      fabric.memoryEndpointRole(endpoint);
  if (!occurrenceRole)
    return invalid("service-leg occurrence endpoint has no memory role");
  const CanonicalServiceEndpointRole effectiveRole =
      *occurrenceRole == FabricMemoryEndpointRole::Manager
          ? CanonicalServiceEndpointRole::Initiate
          : CanonicalServiceEndpointRole::Serve;
  return MemoryEndpointContext{capabilities, effectiveRole};
}

llvm::Expected<std::uint64_t> serviceLegPayloadEnvelope(
    const loom::fabric::CanonicalServiceCapabilityRecord &capability,
    dataflow::StructuralOrdinal legOrdinal) {
  auto roles = dataflow::semantics::getCanonicalServiceLegRoles(
      capability.kind(), legOrdinal);
  if (!roles)
    return roles.takeError();

  std::uint64_t addressBits = 0;
  std::uint64_t dataBits = 0;
  std::uint64_t maskBits = 0;
  if (const auto *addressed =
          std::get_if<loom::fabric::AddressedMemoryCapabilityDomain>(
              &capability.domain())) {
    for (const ::fabric::MemoryAccessClass &access :
         addressed->accesses().accessClasses()) {
      auto envelope = ::fabric::deriveMemoryAccessTransportEnvelope(access);
      if (!envelope)
        return envelope.takeError();
      addressBits = std::max(addressBits, envelope->addressPayloadBits);
      dataBits = std::max(dataBits, envelope->dataPayloadBits);
      maskBits = std::max(maskBits, envelope->maskPayloadBits);
    }
  } else if (!std::holds_alternative<loom::fabric::FenceCapabilityDomain>(
                 capability.domain())) {
    return invalid(
        "message capability has no memory service-leg payload envelope");
  }

  std::uint64_t required = 0;
  for (dataflow::semantics::ServiceValueRole role : *roles) {
    std::uint64_t width = 0;
    switch (role) {
    case dataflow::semantics::ServiceValueRole::Address:
      width = addressBits;
      break;
    case dataflow::semantics::ServiceValueRole::Data:
    case dataflow::semantics::ServiceValueRole::Update:
    case dataflow::semantics::ServiceValueRole::Expected:
    case dataflow::semantics::ServiceValueRole::Desired:
    case dataflow::semantics::ServiceValueRole::Old:
      width = dataBits;
      break;
    case dataflow::semantics::ServiceValueRole::Mask:
      width = maskBits;
      break;
    case dataflow::semantics::ServiceValueRole::Success:
      width = 1;
      break;
    case dataflow::semantics::ServiceValueRole::Control:
    case dataflow::semantics::ServiceValueRole::Completion:
      width = 0;
      break;
    case dataflow::semantics::ServiceValueRole::Payload:
      return invalid("memory service leg contains a message payload role");
    }
    required = std::max(required, width);
  }
  return required;
}

std::vector<std::uint8_t> unsignedBytes(DenseI8ArrayAttr attribute) {
  std::vector<std::uint8_t> result;
  result.reserve(attribute.size());
  for (std::int8_t byte : attribute.asArrayRef())
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

DenseI8ArrayAttr denseBytes(MLIRContext *context,
                            llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  return DenseI8ArrayAttr::get(context, signedBytes);
}

} // namespace

llvm::Error loom::fabric::detail::normalizeSystemServiceLegCarrierAttachments(
    ::fabric::SystemOp root) {
  std::map<AttachmentKey, AttachmentGroup> groups;
  for (auto attachment :
       root.getOps<::fabric::SystemServiceLegCarrierAttachmentOp>()) {
    auto record = decodeServiceLegCarrierAttachmentRecord(
        unsignedBytes(attachment.getRecordAttr()));
    if (!record)
      return record.takeError();
    AttachmentKey key = attachmentKey(*record);
    auto [position, inserted] =
        groups.try_emplace(std::move(key), AttachmentGroup{attachment,
                                                           record->endpoint(),
                                                           record->kind(),
                                                           record->legOrdinal(),
                                                           {},
                                                           {}});
    position->second.carriers.insert(position->second.carriers.end(),
                                     record->carriers().begin(),
                                     record->carriers().end());
    if (!inserted)
      position->second.duplicates.push_back(attachment.getOperation());
  }

  for (auto &[key, group] : groups) {
    (void)key;
    auto record = ServiceLegCarrierAttachmentRecord::create(
        group.endpoint, group.kind, group.legOrdinal,
        std::move(group.carriers));
    if (!record)
      return record.takeError();
    auto bytes = encodeServiceLegCarrierAttachmentRecord(*record);
    if (!bytes)
      return bytes.takeError();
    group.representative.setRecordAttr(denseBytes(root.getContext(), *bytes));
    for (Operation *duplicate : group.duplicates)
      duplicate->erase();
  }
  return llvm::Error::success();
}

llvm::Error loom::fabric::detail::validateSystemServiceLegCarrierAttachments(
    const FabricSystemRootView &system) {
  const FabricArtifactView &fabric = system.artifact();
  std::set<AttachmentKey> expected;
  for (SystemServiceEndpointRef endpoint : fabric.systemServiceEndpoints()) {
    const CanonicalServiceCapabilitySet *capabilities =
        system.serviceEndpointCapabilities(endpoint);
    if (!capabilities ||
        capabilities->plane() != CanonicalServiceEndpointPlane::Memory)
      continue;
    const FabricMemoryEndpointRef memory{
        FabricMemoryEndpointOwnerRef::of(endpoint), 0};
    for (const CanonicalServiceCapabilityRecord &capability :
         capabilities->capabilities()) {
      if (capability.kind() ==
          dataflow::semantics::ServiceKind::MessageTransfer)
        continue;
      const dataflow::StructuralOrdinal legCount =
          dataflow::semantics::getCanonicalServiceLegCount(capability.kind());
      for (dataflow::StructuralOrdinal leg = 0; leg < legCount; ++leg)
        expected.insert({canonicalFabricBytes(memory), capability.kind(), leg});
    }
  }
  for (const FabricSpatialAttachmentRecordView &attachment :
       system.spatialAttachments()) {
    const FabricMemoryEndpointRef *memory = attachment.spatialEndpoint.memory();
    if (!memory)
      continue;
    if (!attachment.serviceEndpoint)
      return invalid("memory spatial attachment has no service endpoint");
    const CanonicalServiceCapabilitySet *capabilities =
        system.serviceEndpointCapabilities(*attachment.serviceEndpoint);
    if (!capabilities ||
        capabilities->plane() != CanonicalServiceEndpointPlane::Memory)
      return invalid(
          "memory spatial attachment has no memory capability authority");
    for (const CanonicalServiceCapabilityRecord &capability :
         capabilities->capabilities()) {
      if (capability.kind() ==
          dataflow::semantics::ServiceKind::MessageTransfer)
        continue;
      const dataflow::StructuralOrdinal legCount =
          dataflow::semantics::getCanonicalServiceLegCount(capability.kind());
      for (dataflow::StructuralOrdinal leg = 0; leg < legCount; ++leg)
        expected.insert(
            {canonicalFabricBytes(*memory), capability.kind(), leg});
    }
  }

  std::set<AttachmentKey> actual;
  for (const ServiceLegCarrierAttachmentRecord &record :
       system.serviceLegCarrierAttachments()) {
    if (llvm::Error error = validateFabricRef(fabric, record.endpoint()))
      return error;
    auto context = resolveMemoryEndpointContext(system, record.endpoint());
    if (!context)
      return context.takeError();
    const CanonicalServiceCapabilitySet *capabilities = context->capabilities;
    const auto capability =
        llvm::find_if(capabilities->capabilities(), [&](const auto &candidate) {
          return candidate.kind() == record.kind();
        });
    if (capability == capabilities->capabilities().end())
      return invalid("service-leg attachment selects an unsupported kind");
    if (record.kind() == dataflow::semantics::ServiceKind::MessageTransfer)
      return invalid("MessageTransfer must not have a service-leg attachment");
    if (record.legOrdinal() >=
        dataflow::semantics::getCanonicalServiceLegCount(record.kind()))
      return invalid("service-leg attachment ordinal is out of range");
    auto legDirection = dataflow::semantics::getCanonicalServiceLegDirection(
        record.kind(), record.legOrdinal());
    if (!legDirection)
      return legDirection.takeError();
    const bool endpointIsInitiator =
        context->effectiveRole == CanonicalServiceEndpointRole::Initiate;
    const bool legSourceIsInitiator =
        *legDirection ==
        dataflow::semantics::ServiceLegDirection::InitiatorToServer;
    const FabricPortDirection expectedDirection =
        endpointIsInitiator == legSourceIsInitiator
            ? FabricPortDirection::Output
            : FabricPortDirection::Input;
    auto requiredPayloadBits =
        serviceLegPayloadEnvelope(*capability, record.legOrdinal());
    if (!requiredPayloadBits)
      return requiredPayloadBits.takeError();
    for (const FabricTransportEndpointRef &carrier : record.carriers()) {
      if (llvm::Error error = validateFabricRef(fabric, carrier))
        return error;
      if (fabric.transportEndpointDirection(carrier) != expectedDirection)
        return invalid("service-leg carrier has the wrong direction");
      const auto dataPath = fabric.transportEndpointDataPath(carrier);
      if (!dataPath || dataPath->payloadWidthBits < *requiredPayloadBits)
        return invalid("service-leg carrier payload is too narrow");
    }
    if (!actual.insert(attachmentKey(record)).second)
      return invalid("service-leg attachment relation repeats one key");
  }

  if (actual != expected)
    return invalid(
        "System does not attach every admitted memory service leg exactly "
        "once");
  return llvm::Error::success();
}
