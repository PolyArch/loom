#ifndef LOOM_FABRIC_IR_SYSTEM_SERVICE_CONTRACT_H
#define LOOM_FABRIC_IR_SYSTEM_SERVICE_CONTRACT_H

#include "Dataflow/IR/DataflowServiceSchema.h"
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

class MessageTransferCapabilityDomain {
public:
  static llvm::Expected<MessageTransferCapabilityDomain>
  create(llvm::ArrayRef<mlir::Type> payloadTypes);
  static llvm::Expected<MessageTransferCapabilityDomain>
  fromCanonical(llvm::ArrayRef<mlir::Type> payloadTypes);

  llvm::ArrayRef<mlir::Type> payloadTypes() const { return payloadTypes_; }

private:
  explicit MessageTransferCapabilityDomain(std::vector<mlir::Type> payloadTypes)
      : payloadTypes_(std::move(payloadTypes)) {}

  std::vector<mlir::Type> payloadTypes_;
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
