#ifndef LOOM_FABRIC_IR_SYSTEM_SERVICE_CONTRACT_H
#define LOOM_FABRIC_IR_SYSTEM_SERVICE_CONTRACT_H

#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Dataflow/IR/DataflowStructuralRefs.h"
#include "Fabric/IR/MemoryActorContractDomain.h"
#include "Fabric/IR/MemoryCapabilityDomains.h"
#include "Fabric/IR/MemoryConsistencyContract.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom::fabric {

/// A field-level refinement of the shared owner catalog. Its persistent bytes
/// are exactly the selected FabricInventoryOwnerRef bytes.
class SystemServiceEndpointOwnerRef {
public:
  static llvm::Expected<SystemServiceEndpointOwnerRef>
  create(FabricInventoryOwnerRef owner);

  const FabricInventoryOwnerRef &owner() const { return owner_; }

private:
  explicit SystemServiceEndpointOwnerRef(FabricInventoryOwnerRef owner)
      : owner_(std::move(owner)) {}

  FabricInventoryOwnerRef owner_;
};

std::vector<std::uint8_t> encodeSystemServiceEndpointOwnerRef(
    const SystemServiceEndpointOwnerRef &reference);
llvm::Expected<SystemServiceEndpointOwnerRef>
decodeSystemServiceEndpointOwnerRef(llvm::ArrayRef<std::uint8_t> bytes);

enum class CanonicalServiceEndpointRole : std::uint32_t {
  Initiate,
  Serve,
};

enum class CanonicalServiceEndpointPlane : std::uint32_t {
  Transport,
  Memory,
};

using ServiceProgress =
    std::variant<::fabric::BoundedCompletion, ::fabric::FairEventual>;

/// One rate, capacity, and progress guarantee. Whether the rate is an issue or
/// accept rate is derived from the capability's endpoint role.
class ServiceRateContractRecord {
public:
  static llvm::Expected<ServiceRateContractRecord>
  create(ClockDomainRef rateClock, std::uint64_t operationsPerWindow,
         std::uint64_t windowTicks, std::uint64_t maxOutstanding,
         ServiceProgress progress);

  const ClockDomainRef &rateClock() const { return rateClock_; }
  std::uint64_t operationsPerWindow() const { return operationsPerWindow_; }
  std::uint64_t windowTicks() const { return windowTicks_; }
  std::uint64_t maxOutstanding() const { return maxOutstanding_; }
  const ServiceProgress &progress() const { return progress_; }

private:
  ServiceRateContractRecord(ClockDomainRef rateClock,
                            std::uint64_t operationsPerWindow,
                            std::uint64_t windowTicks,
                            std::uint64_t maxOutstanding,
                            ServiceProgress progress)
      : rateClock_(std::move(rateClock)),
        operationsPerWindow_(operationsPerWindow), windowTicks_(windowTicks),
        maxOutstanding_(maxOutstanding), progress_(std::move(progress)) {}

  ClockDomainRef rateClock_;
  std::uint64_t operationsPerWindow_;
  std::uint64_t windowTicks_;
  std::uint64_t maxOutstanding_;
  ServiceProgress progress_;
};

class FixedVectorMessagePayloadDomain final {
public:
  static llvm::Expected<FixedVectorMessagePayloadDomain>
  create(llvm::ArrayRef<mlir::Type> elementTypes,
         std::uint64_t maximumPayloadBits, std::uint32_t maximumRank);
  static llvm::Expected<FixedVectorMessagePayloadDomain>
  fromCanonical(llvm::ArrayRef<mlir::Type> elementTypes,
                std::uint64_t maximumPayloadBits, std::uint32_t maximumRank);

  llvm::ArrayRef<mlir::Type> elementTypes() const { return elementTypes_; }
  std::uint64_t maximumPayloadBits() const { return maximumPayloadBits_; }
  std::uint32_t maximumRank() const { return maximumRank_; }
  llvm::Expected<bool> admits(mlir::Type payloadType) const;

private:
  FixedVectorMessagePayloadDomain(
      std::vector<mlir::Type> elementTypes,
      std::vector<std::vector<std::uint8_t>> canonicalElementTypes,
      std::uint64_t maximumPayloadBits, std::uint32_t maximumRank)
      : elementTypes_(std::move(elementTypes)),
        canonicalElementTypes_(std::move(canonicalElementTypes)),
        maximumPayloadBits_(maximumPayloadBits), maximumRank_(maximumRank) {}

  std::vector<mlir::Type> elementTypes_;
  std::vector<std::vector<std::uint8_t>> canonicalElementTypes_;
  std::uint64_t maximumPayloadBits_;
  std::uint32_t maximumRank_;
};

class MessageTransferCapabilityDomain {
public:
  static llvm::Expected<MessageTransferCapabilityDomain>
  create(llvm::ArrayRef<mlir::Type> payloadTypes,
         std::optional<FixedVectorMessagePayloadDomain> fixedVectors =
             std::nullopt,
         ::fabric::PointerFormatRelation pointerFormats = {});
  static llvm::Expected<MessageTransferCapabilityDomain>
  fromCanonical(llvm::ArrayRef<mlir::Type> payloadTypes,
                std::optional<FixedVectorMessagePayloadDomain> fixedVectors =
                    std::nullopt,
                ::fabric::PointerFormatRelation pointerFormats = {});

  llvm::ArrayRef<mlir::Type> payloadTypes() const { return payloadTypes_; }
  const std::optional<FixedVectorMessagePayloadDomain> &fixedVectors() const {
    return fixedVectors_;
  }
  const ::fabric::PointerFormatRelation &pointerFormats() const {
    return pointerFormats_;
  }
  llvm::Expected<bool>
  admits(mlir::Type payloadType,
         const ::loom::PointerLayout *pointerLayout = nullptr) const;

private:
  MessageTransferCapabilityDomain(
      std::vector<mlir::Type> payloadTypes,
      std::vector<std::vector<std::uint8_t>> canonicalPayloadTypes,
      std::optional<FixedVectorMessagePayloadDomain> fixedVectors,
      ::fabric::PointerFormatRelation pointerFormats)
      : payloadTypes_(std::move(payloadTypes)),
        canonicalPayloadTypes_(std::move(canonicalPayloadTypes)),
        fixedVectors_(std::move(fixedVectors)),
        pointerFormats_(std::move(pointerFormats)) {}

  std::vector<mlir::Type> payloadTypes_;
  std::vector<std::vector<std::uint8_t>> canonicalPayloadTypes_;
  std::optional<FixedVectorMessagePayloadDomain> fixedVectors_;
  ::fabric::PointerFormatRelation pointerFormats_;
};

class AddressedMemoryCapabilityDomain {
public:
  static llvm::Expected<AddressedMemoryCapabilityDomain>
  create(::fabric::MemoryActorContractDomain actorContracts,
         ::fabric::ParameterizedMemoryAccessDomain accesses,
         ::fabric::UnsignedDomain addressBytes,
         std::uint64_t serviceBeatWidthBits,
         std::optional<MemoryConsistencyDomainRef> consistencyDomain);

  const ::fabric::MemoryActorContractDomain &actorContracts() const {
    return actorContracts_;
  }
  const ::fabric::ParameterizedMemoryAccessDomain &accesses() const {
    return accesses_;
  }
  const ::fabric::UnsignedDomain &addressBytes() const { return addressBytes_; }
  std::uint64_t serviceBeatWidthBits() const { return serviceBeatWidthBits_; }
  const std::optional<MemoryConsistencyDomainRef> &consistencyDomain() const {
    return consistencyDomain_;
  }

private:
  AddressedMemoryCapabilityDomain(
      ::fabric::MemoryActorContractDomain actorContracts,
      ::fabric::ParameterizedMemoryAccessDomain accesses,
      ::fabric::UnsignedDomain addressBytes, std::uint64_t serviceBeatWidthBits,
      std::optional<MemoryConsistencyDomainRef> consistencyDomain)
      : actorContracts_(std::move(actorContracts)),
        accesses_(std::move(accesses)), addressBytes_(std::move(addressBytes)),
        serviceBeatWidthBits_(serviceBeatWidthBits),
        consistencyDomain_(std::move(consistencyDomain)) {}

  ::fabric::MemoryActorContractDomain actorContracts_;
  ::fabric::ParameterizedMemoryAccessDomain accesses_;
  ::fabric::UnsignedDomain addressBytes_;
  std::uint64_t serviceBeatWidthBits_;
  std::optional<MemoryConsistencyDomainRef> consistencyDomain_;
};

class FenceCapabilityDomain {
public:
  static llvm::Expected<FenceCapabilityDomain>
  create(::fabric::MemoryActorContractDomain actorContracts,
         MemoryConsistencyDomainRef consistencyDomain);

  const ::fabric::MemoryActorContractDomain &actorContracts() const {
    return actorContracts_;
  }
  const MemoryConsistencyDomainRef &consistencyDomain() const {
    return consistencyDomain_;
  }

private:
  FenceCapabilityDomain(::fabric::MemoryActorContractDomain actorContracts,
                        MemoryConsistencyDomainRef consistencyDomain)
      : actorContracts_(std::move(actorContracts)),
        consistencyDomain_(std::move(consistencyDomain)) {}

  ::fabric::MemoryActorContractDomain actorContracts_;
  MemoryConsistencyDomainRef consistencyDomain_;
};

using CanonicalServiceCapabilityDomain =
    std::variant<MessageTransferCapabilityDomain,
                 AddressedMemoryCapabilityDomain, FenceCapabilityDomain>;

class CanonicalServiceCapabilityRecord {
public:
  static llvm::Expected<CanonicalServiceCapabilityRecord> create(
      dataflow::semantics::ServiceKind kind, CanonicalServiceEndpointRole role,
      CanonicalServiceCapabilityDomain domain, ServiceRateContractRecord rate);

  dataflow::semantics::ServiceKind kind() const { return kind_; }
  CanonicalServiceEndpointRole role() const { return role_; }
  const CanonicalServiceCapabilityDomain &domain() const { return domain_; }
  const ServiceRateContractRecord &rate() const { return rate_; }

private:
  CanonicalServiceCapabilityRecord(dataflow::semantics::ServiceKind kind,
                                   CanonicalServiceEndpointRole role,
                                   CanonicalServiceCapabilityDomain domain,
                                   ServiceRateContractRecord rate)
      : kind_(kind), role_(role), domain_(std::move(domain)),
        rate_(std::move(rate)) {}

  dataflow::semantics::ServiceKind kind_;
  CanonicalServiceEndpointRole role_;
  CanonicalServiceCapabilityDomain domain_;
  ServiceRateContractRecord rate_;
};

class CanonicalServiceCapabilitySet {
public:
  static llvm::Expected<CanonicalServiceCapabilitySet>
  create(std::vector<CanonicalServiceCapabilityRecord> capabilities);
  static llvm::Expected<CanonicalServiceCapabilitySet>
  fromCanonical(std::vector<CanonicalServiceCapabilityRecord> capabilities);

  llvm::ArrayRef<CanonicalServiceCapabilityRecord> capabilities() const {
    return capabilities_;
  }
  CanonicalServiceEndpointRole role() const {
    return capabilities_.front().role();
  }
  CanonicalServiceEndpointPlane plane() const;

private:
  explicit CanonicalServiceCapabilitySet(
      std::vector<CanonicalServiceCapabilityRecord> capabilities)
      : capabilities_(std::move(capabilities)) {}

  std::vector<CanonicalServiceCapabilityRecord> capabilities_;
};

llvm::Expected<std::vector<std::uint8_t>> encodeCanonicalServiceCapabilitySet(
    const CanonicalServiceCapabilitySet &capabilities);
llvm::Expected<CanonicalServiceCapabilitySet>
decodeCanonicalServiceCapabilitySet(llvm::ArrayRef<std::uint8_t> bytes,
                                    mlir::MLIRContext *context);

/// One Fabric-owned candidate-carrier relation for a canonical memory-service
/// leg. Service schema semantics remain owned by Dataflow.
class ServiceLegCarrierAttachmentRecord {
public:
  static llvm::Expected<ServiceLegCarrierAttachmentRecord>
  create(FabricMemoryEndpointRef endpoint,
         dataflow::semantics::ServiceKind kind,
         dataflow::StructuralOrdinal legOrdinal,
         std::vector<FabricTransportEndpointRef> carriers);
  static llvm::Expected<ServiceLegCarrierAttachmentRecord>
  fromCanonical(FabricMemoryEndpointRef endpoint,
                dataflow::semantics::ServiceKind kind,
                dataflow::StructuralOrdinal legOrdinal,
                std::vector<FabricTransportEndpointRef> carriers);

  const FabricMemoryEndpointRef &endpoint() const { return endpoint_; }
  dataflow::semantics::ServiceKind kind() const { return kind_; }
  dataflow::StructuralOrdinal legOrdinal() const { return legOrdinal_; }
  llvm::ArrayRef<FabricTransportEndpointRef> carriers() const {
    return carriers_;
  }

private:
  ServiceLegCarrierAttachmentRecord(
      FabricMemoryEndpointRef endpoint, dataflow::semantics::ServiceKind kind,
      dataflow::StructuralOrdinal legOrdinal,
      std::vector<FabricTransportEndpointRef> carriers)
      : endpoint_(std::move(endpoint)), kind_(kind), legOrdinal_(legOrdinal),
        carriers_(std::move(carriers)) {}

  FabricMemoryEndpointRef endpoint_;
  dataflow::semantics::ServiceKind kind_;
  dataflow::StructuralOrdinal legOrdinal_;
  std::vector<FabricTransportEndpointRef> carriers_;
};

llvm::Expected<std::vector<std::uint8_t>>
encodeServiceLegCarrierAttachmentRecord(
    const ServiceLegCarrierAttachmentRecord &record);
llvm::Expected<ServiceLegCarrierAttachmentRecord>
decodeServiceLegCarrierAttachmentRecord(llvm::ArrayRef<std::uint8_t> bytes);

struct AddressOffsetTransform {
  std::uint32_t addressWidth = 0;
  std::int64_t signedOffset = 0;
};

struct AddressMaskXorTransform {
  std::uint32_t addressWidth = 0;
  std::uint64_t andMask = 0;
  std::uint64_t xorMask = 0;
};

struct StaticInterleaveTransform {
  std::uint64_t granuleBytes = 0;
  std::uint64_t outputCount = 0;
};

struct CoherentMemoryRegionCorrespondence {
  FabricMemoryServiceRegionRef input;
  FabricMemoryServiceRegionRef output;
};

inline bool operator==(const CoherentMemoryRegionCorrespondence &left,
                       const CoherentMemoryRegionCorrespondence &right) {
  return left.input == right.input && left.output == right.output;
}

struct CoherentMemoryTransform {
  MemoryConsistencyDomainRef consistencyDomain;
  std::vector<CoherentMemoryRegionCorrespondence> regions;
};

using ServiceTransformContract =
    std::variant<AddressOffsetTransform, AddressMaskXorTransform,
                 StaticInterleaveTransform, CoherentMemoryTransform>;

class SystemServiceTransformRecord {
public:
  static llvm::Expected<SystemServiceTransformRecord>
  create(std::vector<FabricMemoryEndpointRef> inputs,
         std::vector<FabricMemoryEndpointRef> outputs,
         ServiceTransformContract contract);

  llvm::ArrayRef<FabricMemoryEndpointRef> inputs() const { return inputs_; }
  llvm::ArrayRef<FabricMemoryEndpointRef> outputs() const { return outputs_; }
  const ServiceTransformContract &contract() const { return contract_; }

private:
  SystemServiceTransformRecord(std::vector<FabricMemoryEndpointRef> inputs,
                               std::vector<FabricMemoryEndpointRef> outputs,
                               ServiceTransformContract contract)
      : inputs_(std::move(inputs)), outputs_(std::move(outputs)),
        contract_(std::move(contract)) {}

  std::vector<FabricMemoryEndpointRef> inputs_;
  std::vector<FabricMemoryEndpointRef> outputs_;
  ServiceTransformContract contract_;
};

llvm::Expected<std::vector<std::uint8_t>>
encodeSystemServiceTransformRecord(const SystemServiceTransformRecord &record);
llvm::Expected<SystemServiceTransformRecord>
decodeSystemServiceTransformRecord(llvm::ArrayRef<std::uint8_t> bytes);

} // namespace loom::fabric

#endif // LOOM_FABRIC_IR_SYSTEM_SERVICE_CONTRACT_H
