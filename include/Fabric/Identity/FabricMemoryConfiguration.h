#ifndef LOOM_FABRIC_IDENTITY_FABRICMEMORYCONFIGURATION_H
#define LOOM_FABRIC_IDENTITY_FABRICMEMORYCONFIGURATION_H

#include "Common/Artifact.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/IR/MemoryConnectivityContract.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom::fabric {

class FabricArtifactView;

struct FabricMemoryExternalRoleSource final {
  FabricOrdinal endpoint = 0;
  llvm::APInt tag = llvm::APInt(1, 0);

  friend bool operator==(const FabricMemoryExternalRoleSource &lhs,
                         const FabricMemoryExternalRoleSource &rhs) {
    return lhs.endpoint == rhs.endpoint && lhs.tag == rhs.tag;
  }
};

struct FabricMemoryInternalRoleSource final {
  FabricOrdinal connection = 0;

  friend bool operator==(FabricMemoryInternalRoleSource lhs,
                         FabricMemoryInternalRoleSource rhs) {
    return lhs.connection == rhs.connection;
  }
};

using FabricMemoryRoleSource = std::variant<FabricMemoryExternalRoleSource,
                                            FabricMemoryInternalRoleSource>;

struct FabricMemoryRoleDestination final {
  std::optional<FabricMemoryExternalRoleSource> external;
  std::vector<FabricOrdinal> internalConnections;

  friend bool operator==(const FabricMemoryRoleDestination &lhs,
                         const FabricMemoryRoleDestination &rhs) {
    return lhs.external == rhs.external &&
           lhs.internalConnections == rhs.internalConnections;
  }
};

/// One exact point in a capability alternative's finite actor-contract
/// product. Clause and value ordinals index the canonical Fabric-owned
/// domains; they do not introduce another enum or contract codec.
struct FabricMemoryActorContractSelection final {
  FabricOrdinal clause = 0;
  std::vector<FabricOrdinal> values;

  friend bool operator==(const FabricMemoryActorContractSelection &lhs,
                         const FabricMemoryActorContractSelection &rhs) {
    return lhs.clause == rhs.clause && lhs.values == rhs.values;
  }
};

/// One exact point in a capability alternative's parameterized access
/// relation. Finite members use canonical ordinals; interval members retain
/// their exact unsigned value.
struct FabricMemoryAccessSelection final {
  FabricOrdinal accessClass = 0;
  std::uint64_t elementWidthBits = 0;
  std::uint64_t flattenedLaneCount = 0;
  FabricOrdinal maskInactivePair = 0;
  std::uint64_t sourceAlignmentBytes = 0;
  std::uint32_t addressLaneWidthBits = 0;
  FabricOrdinal addressPointerFormat = 0;
  std::optional<FabricOrdinal> dataPointerFormat;

  friend bool operator==(const FabricMemoryAccessSelection &lhs,
                         const FabricMemoryAccessSelection &rhs) {
    return lhs.accessClass == rhs.accessClass &&
           lhs.elementWidthBits == rhs.elementWidthBits &&
           lhs.flattenedLaneCount == rhs.flattenedLaneCount &&
           lhs.maskInactivePair == rhs.maskInactivePair &&
           lhs.sourceAlignmentBytes == rhs.sourceAlignmentBytes &&
           lhs.addressLaneWidthBits == rhs.addressLaneWidthBits &&
           lhs.addressPointerFormat == rhs.addressPointerFormat &&
           lhs.dataPointerFormat == rhs.dataPointerFormat;
  }
};

struct FabricMemoryOperationRow final {
  FabricOrdinal physicalPort = 0;
  FabricOrdinal capabilityAlternative = 0;
  FabricOrdinal usePattern = 0;
  FabricMemoryActorContractSelection actorContract;
  std::optional<FabricMemoryAccessSelection> access;
  std::uint64_t baseAddressBytes = 0;
  std::vector<std::optional<FabricMemoryRoleSource>> roleSources;
  std::vector<std::optional<FabricMemoryRoleDestination>> roleDestinations;
  ::fabric::MemoryDispatchTarget serviceTarget;

  friend bool operator==(const FabricMemoryOperationRow &lhs,
                         const FabricMemoryOperationRow &rhs) {
    return lhs.physicalPort == rhs.physicalPort &&
           lhs.capabilityAlternative == rhs.capabilityAlternative &&
           lhs.usePattern == rhs.usePattern &&
           lhs.actorContract == rhs.actorContract && lhs.access == rhs.access &&
           lhs.baseAddressBytes == rhs.baseAddressBytes &&
           lhs.roleSources == rhs.roleSources &&
           lhs.roleDestinations == rhs.roleDestinations &&
           lhs.serviceTarget == rhs.serviceTarget;
  }
};

struct FabricMemoryRangeMatch final {
  std::uint64_t base = 0;
  std::uint64_t size = 0;

  friend bool operator==(FabricMemoryRangeMatch lhs,
                         FabricMemoryRangeMatch rhs) {
    return lhs.base == rhs.base && lhs.size == rhs.size;
  }
};

struct FabricMemoryPrefixMatch final {
  std::uint64_t value = 0;
  std::uint8_t prefixLength = 0;

  friend bool operator==(FabricMemoryPrefixMatch lhs,
                         FabricMemoryPrefixMatch rhs) {
    return lhs.value == rhs.value && lhs.prefixLength == rhs.prefixLength;
  }
};

struct FabricMemoryAddressSpaceMatch final {
  std::uint32_t addressSpace = 0;

  friend bool operator==(FabricMemoryAddressSpaceMatch lhs,
                         FabricMemoryAddressSpaceMatch rhs) {
    return lhs.addressSpace == rhs.addressSpace;
  }
};

struct FabricMemoryContextMatch final {
  std::uint64_t context = 0;

  friend bool operator==(FabricMemoryContextMatch lhs,
                         FabricMemoryContextMatch rhs) {
    return lhs.context == rhs.context;
  }
};

using FabricMemoryProviderMatch =
    std::variant<FabricMemoryRangeMatch, FabricMemoryPrefixMatch,
                 FabricMemoryAddressSpaceMatch, FabricMemoryContextMatch>;

struct FabricMemoryProviderDecodeRow final {
  std::vector<FabricMemoryProviderMatch> matches;
  ::fabric::MemoryDispatchTarget serviceTarget;
  std::uint64_t baseOffsetBytes = 0;

  friend bool operator==(const FabricMemoryProviderDecodeRow &lhs,
                         const FabricMemoryProviderDecodeRow &rhs) {
    return lhs.matches == rhs.matches &&
           lhs.serviceTarget == rhs.serviceTarget &&
           lhs.baseOffsetBytes == rhs.baseOffsetBytes;
  }
};

struct FabricMemoryDisabled final {
  friend bool operator==(FabricMemoryDisabled, FabricMemoryDisabled) {
    return true;
  }
};

struct FabricMemoryActive final {
  std::vector<std::optional<FabricMemoryOperationRow>> operationRows;
  std::vector<std::vector<std::optional<FabricMemoryProviderDecodeRow>>>
      providerDecodeRows;

  friend bool operator==(const FabricMemoryActive &lhs,
                         const FabricMemoryActive &rhs) {
    return lhs.operationRows == rhs.operationRows &&
           lhs.providerDecodeRows == rhs.providerDecodeRows;
  }
};

using FabricMemoryConfigurationValue =
    std::variant<FabricMemoryDisabled, FabricMemoryActive>;

struct FabricMemoryOperationRowLayout final {
  std::uint64_t bitOffset = 0;
  std::uint64_t bitCount = 0;
  std::uint64_t physicalPortOffset = 0;
  std::uint64_t capabilityOffset = 0;
  std::uint64_t usePatternOffset = 0;
  std::uint64_t actorClauseOffset = 0;
  std::vector<std::uint64_t> actorValueOffsets;
  std::uint64_t accessPresentOffset = 0;
  std::uint64_t accessClassOffset = 0;
  std::uint64_t elementWidthOffset = 0;
  std::uint64_t laneCountOffset = 0;
  std::uint64_t maskPairOffset = 0;
  std::uint64_t alignmentOffset = 0;
  std::uint64_t addressLaneWidthOffset = 0;
  std::uint64_t addressPointerFormatOffset = 0;
  std::uint64_t dataPointerPresentOffset = 0;
  std::uint64_t dataPointerFormatOffset = 0;
  std::uint64_t baseAddressOffset = 0;
  std::vector<std::uint64_t> roleSourceOffsets;
  std::vector<std::uint64_t> roleDestinationOffsets;
  std::uint64_t serviceTargetOffset = 0;
};

struct FabricMemoryProviderRowLayout final {
  std::uint64_t bitOffset = 0;
  std::uint64_t bitCount = 0;
  std::vector<std::uint64_t> matchOffsets;
  std::uint64_t serviceTargetOffset = 0;
  std::uint64_t baseOffsetOffset = 0;
};

/// Fixed-capacity direct-carrier layout for one memory occurrence. Every
/// width is derived from the exact immutable Fabric inventories and domains.
struct FabricMemoryConfigurationLayout final {
  std::optional<::fabric::Schedule> schedule;
  std::uint32_t roleCount = 0;
  std::uint32_t operationRowCount = 0;
  std::uint32_t physicalPortCount = 0;
  std::uint32_t transportEndpointCount = 0;
  std::uint32_t internalConnectionCount = 0;
  std::uint32_t managerEndpointCount = 0;
  std::uint32_t tagWidthBits = 0;
  std::uint32_t physicalPortBitCount = 0;
  std::uint32_t capabilityBitCount = 0;
  std::uint32_t usePatternBitCount = 0;
  std::uint32_t actorClauseBitCount = 0;
  std::vector<std::uint32_t> actorValueBitCounts;
  std::uint32_t accessClassBitCount = 0;
  std::uint32_t maskPairBitCount = 0;
  std::uint32_t pointerFormatBitCount = 0;
  std::uint32_t transportEndpointBitCount = 0;
  std::uint32_t internalConnectionBitCount = 0;
  std::uint32_t serviceTargetBitCount = 0;
  std::uint64_t roleSourceBitCount = 0;
  std::uint64_t roleDestinationBitCount = 0;
  std::uint64_t carrierBitCount = 0;
  std::vector<FabricMemoryOperationRowLayout> operationRows;
  std::vector<std::vector<FabricMemoryProviderRowLayout>> providerRows;
};

class FabricMemoryConfigurationSchemaView final {
public:
  FabricMemoryOccurrenceRef memory() const { return memory_; }
  const FabricSemanticConfigFieldRef &field() const { return field_; }
  const FabricMemoryConfigurationLayout &layout() const { return layout_; }

  llvm::Expected<CanonicalSemanticBytes>
  encode(const FabricMemoryConfigurationValue &value) const;

  llvm::Expected<FabricMemoryConfigurationValue>
  decode(llvm::ArrayRef<std::uint8_t> bytes) const;

private:
  FabricMemoryConfigurationSchemaView(const FabricArtifactView *fabric,
                                      FabricMemoryOccurrenceRef memory,
                                      FabricSemanticConfigFieldRef field,
                                      FabricMemoryConfigurationLayout layout)
      : fabric_(fabric), memory_(memory), field_(std::move(field)),
        layout_(std::move(layout)) {}

  const FabricArtifactView *fabric_ = nullptr;
  FabricMemoryOccurrenceRef memory_;
  FabricSemanticConfigFieldRef field_;
  FabricMemoryConfigurationLayout layout_;

  friend class FabricArtifactView;
};

} // namespace loom::fabric

#endif // LOOM_FABRIC_IDENTITY_FABRICMEMORYCONFIGURATION_H
