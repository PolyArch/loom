#ifndef LOOM_FABRIC_IR_MEMORY_OPERATION_PORT_H
#define LOOM_FABRIC_IR_MEMORY_OPERATION_PORT_H

#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/IR/FabricEnums.h"
#include "Fabric/IR/MemoryActorContractDomain.h"
#include "Fabric/IR/MemoryCapabilityDomains.h"
#include "Fabric/IR/MemoryPortTransaction.h"
#include "Fabric/IR/ResourceContract.h"
#include "Fabric/Identity/FabricRefs.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace fabric {

inline constexpr char kMemoryOperationPortsAttrName[] =
    "memory_operation_ports";

/// One transient token endpoint descriptor derived from a fabric.mem function
/// type. Ordinals are assigned to token inputs first and token outputs second;
/// memref capability endpoints remain in their separate inventory.
struct MemoryTransportEndpointDescriptor {
  loom::fabric::FabricPortDirection direction;
  std::uint32_t payloadWidth;
  std::optional<std::uint32_t> tagWidth;
};

llvm::Expected<std::vector<MemoryTransportEndpointDescriptor>>
deriveMemoryTransportEndpointInventory(mlir::FunctionType functionType);

struct MemoryRoleEndpointBindingRecord {
  dataflow::semantics::ServiceValueRole role;
  std::uint64_t endpointOrdinal;
};

struct MemoryOperationPatternRecord {
  MemoryPortTransactionProjection transactionProjection;
};

struct MemoryCapabilityAlternativeRecord {
  MemoryActorContractDomain actorContractDomain;
  std::vector<MemoryRoleEndpointBindingRecord> roleToEndpoint;
  std::optional<ParameterizedMemoryAccessDomain> accessDomain;
  std::vector<UsePatternKey> admissibleUsePatterns;
};

struct MemoryOperationPortDeclaration {
  std::vector<std::uint64_t> endpointInventory;
  ResourceContract resourceContract;
  std::vector<MemoryOperationPatternRecord> operationPatternSemantics;
  std::vector<MemoryCapabilityAlternativeRecord> capabilityAlternatives;
};

struct MemoryCapabilityMatch {
  std::uint64_t alternativeOrdinal;
  std::vector<UsePatternKey> admissibleUsePatterns;
};

/// One complete canonical persistent memory-operation port record. The
/// containing fabric.mem owns its schedule and function type; this record
/// references that token endpoint inventory without repeating endpoint types.
class MemoryOperationPortRecord {
public:
  /// Normalizes authoring alternatives into the unique reduced relation and
  /// validates the complete physical capability against its owner.
  static llvm::Expected<MemoryOperationPortRecord>
  create(mlir::MLIRContext *context, Schedule schedule,
         llvm::ArrayRef<MemoryTransportEndpointDescriptor> endpoints,
         MemoryOperationPortDeclaration declaration);

  /// Strictly imports an already canonical declaration. Reordered, split, or
  /// otherwise equivalent noncanonical declarations are rejected.
  static llvm::Expected<MemoryOperationPortRecord>
  fromCanonical(mlir::MLIRContext *context, Schedule schedule,
                llvm::ArrayRef<MemoryTransportEndpointDescriptor> endpoints,
                MemoryOperationPortDeclaration declaration);

  llvm::ArrayRef<std::uint64_t> endpointInventory() const {
    return declaration_.endpointInventory;
  }
  const ResourceContract &resourceContract() const {
    return declaration_.resourceContract;
  }
  llvm::ArrayRef<MemoryOperationPatternRecord> operationPatterns() const {
    return declaration_.operationPatternSemantics;
  }
  llvm::ArrayRef<MemoryCapabilityAlternativeRecord>
  capabilityAlternatives() const {
    return declaration_.capabilityAlternatives;
  }

  /// Returns every canonical capability alternative that realizes one exact
  /// actor, retaining the alternative ordinal that owns its role binding. An
  /// empty result is ordinary Mapping infeasibility; malformed or mutually
  /// inconsistent projections are errors.
  llvm::Expected<std::vector<MemoryCapabilityMatch>> matchingCapabilities(
      const dataflow::CanonicalActorSchemaProjection &actor,
      const dataflow::semantics::CanonicalService &service,
      const std::optional<dataflow::semantics::CanonicalMemoryAccessView>
          &access) const;

private:
  MemoryOperationPortRecord(
      std::vector<MemoryTransportEndpointDescriptor> endpoints,
      MemoryOperationPortDeclaration declaration)
      : endpoints_(std::move(endpoints)), declaration_(std::move(declaration)) {
  }

  std::vector<MemoryTransportEndpointDescriptor> endpoints_;
  MemoryOperationPortDeclaration declaration_;
};

llvm::Expected<std::vector<std::uint8_t>>
encodeMemoryOperationPortRecord(const MemoryOperationPortRecord &record);

llvm::Expected<MemoryOperationPortRecord> decodeMemoryOperationPortRecord(
    llvm::ArrayRef<std::uint8_t> bytes, mlir::MLIRContext *context,
    Schedule schedule,
    llvm::ArrayRef<MemoryTransportEndpointDescriptor> endpoints);

/// Strictly imports one complete canonical operation-port inventory from its
/// Fabric IR carrier. Every array element is one DenseI8ArrayAttr containing
/// the exact owner wire record; MLIR attribute structure is only framing.
llvm::Expected<std::vector<MemoryOperationPortRecord>>
decodeMemoryOperationPortInventory(
    mlir::ArrayAttr records, mlir::MLIRContext *context, Schedule schedule,
    llvm::ArrayRef<MemoryTransportEndpointDescriptor> endpoints);

} // namespace fabric

#endif // LOOM_FABRIC_IR_MEMORY_OPERATION_PORT_H
