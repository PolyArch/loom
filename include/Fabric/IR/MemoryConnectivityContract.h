#ifndef LOOM_FABRIC_IR_MEMORY_CONNECTIVITY_CONTRACT_H
#define LOOM_FABRIC_IR_MEMORY_CONNECTIVITY_CONTRACT_H

#include "Fabric/IR/MemoryOperationPort.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

namespace fabric {

struct LocalMemoryDispatchTarget {};

inline bool operator==(LocalMemoryDispatchTarget, LocalMemoryDispatchTarget) {
  return true;
}

struct ManagerMemoryDispatchTarget {
  std::uint64_t endpointOrdinal = 0;
};

inline bool operator==(ManagerMemoryDispatchTarget left,
                       ManagerMemoryDispatchTarget right) {
  return left.endpointOrdinal == right.endpointOrdinal;
}

using MemoryDispatchTarget =
    std::variant<LocalMemoryDispatchTarget, ManagerMemoryDispatchTarget>;

enum class MemoryProviderMatchField : std::uint32_t {
  Range,
  Prefix,
  AddressSpace,
  Context,
};

enum class MemoryProviderAddressTransform : std::uint32_t {
  None,
  ConstantBaseOffset,
};

struct MemoryOperationPortDispatchDeclaration {
  std::vector<std::vector<MemoryDispatchTarget>> capabilityTargetDomains;
};

struct MemorySubordinateDispatchDeclaration {
  std::uint64_t maxExposedBindings = 0;
  std::vector<MemoryProviderMatchField> matchFields;
  MemoryProviderAddressTransform addressTransform =
      MemoryProviderAddressTransform::None;
  std::vector<MemoryDispatchTarget> targetDomain;
};

struct MemoryInternalConnectionDeclaration {
  std::uint64_t sourceEndpointOrdinal = 0;
  std::uint64_t sinkEndpointOrdinal = 0;
};

struct MemoryConnectivityDeclaration {
  std::vector<MemoryOperationPortDispatchDeclaration> operationPorts;
  std::vector<MemorySubordinateDispatchDeclaration> subordinateEndpoints;
  std::vector<MemoryInternalConnectionDeclaration> internalConnections;
};

/// The complete occurrence-owned fixed memory connectivity relation. Source
/// identity is positional in the operation-port, capability-alternative, and
/// subordinate endpoint inventories; no persistent dense ID is introduced.
class MemoryConnectivityContractRecord final {
public:
  static llvm::Expected<MemoryConnectivityContractRecord>
  create(MemoryConnectivityDeclaration declaration);

  static llvm::Expected<MemoryConnectivityContractRecord>
  fromCanonical(MemoryConnectivityDeclaration declaration);

  llvm::ArrayRef<MemoryOperationPortDispatchDeclaration>
  operationPorts() const {
    return declaration_.operationPorts;
  }
  llvm::ArrayRef<MemorySubordinateDispatchDeclaration>
  subordinateEndpoints() const {
    return declaration_.subordinateEndpoints;
  }
  llvm::ArrayRef<MemoryInternalConnectionDeclaration>
  internalConnections() const {
    return declaration_.internalConnections;
  }

private:
  explicit MemoryConnectivityContractRecord(
      MemoryConnectivityDeclaration declaration)
      : declaration_(std::move(declaration)) {}

  MemoryConnectivityDeclaration declaration_;
};

llvm::Expected<std::vector<std::uint8_t>>
encodeMemoryConnectivityContractRecord(
    const MemoryConnectivityContractRecord &record);

llvm::Expected<MemoryConnectivityContractRecord>
decodeMemoryConnectivityContractRecord(llvm::ArrayRef<std::uint8_t> bytes);

/// Validates one canonical record against the exact occurrence inventories.
/// This is the sole owner check for H_dispatch and internal token eligibility.
llvm::Error validateMemoryConnectivityContract(
    const MemoryConnectivityContractRecord &record,
    llvm::ArrayRef<MemoryOperationPortRecord> operationPorts,
    llvm::ArrayRef<MemoryTransportEndpointDescriptor> transportEndpoints,
    std::uint64_t managerEndpointCount, std::uint64_t subordinateEndpointCount,
    bool hasLocalMemoryService);

} // namespace fabric

#endif // LOOM_FABRIC_IR_MEMORY_CONNECTIVITY_CONTRACT_H
